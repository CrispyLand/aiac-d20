package com.crispyland.agent;

import com.crispyland.agent.invariant.Invariant;
import com.crispyland.agent.invariant.InvariantGuard;
import com.crispyland.agent.invariant.InvariantStore;
import com.crispyland.agent.invariant.Invariants;
import com.crispyland.agent.judge.Judge;
import com.crispyland.agent.judge.Verdict;
import com.crispyland.agent.llm.ChatRequest;
import com.crispyland.agent.llm.ChatResponse;
import com.crispyland.agent.llm.LlmClient;
import com.crispyland.agent.llm.ToolBox;
import com.crispyland.agent.llm.ToolCall;
import com.crispyland.agent.llm.ToolResult;
import com.crispyland.agent.llm.ToolRound;
import com.crispyland.agent.llm.ToolSpec;
import com.crispyland.agent.memory.ConversationStore;
import com.crispyland.agent.memory.Facts;
import com.crispyland.agent.memory.HistoryCompressor;
import com.crispyland.agent.memory.LongTermKind;
import com.crispyland.agent.memory.LongTermMemory;
import com.crispyland.agent.memory.LongTermStore;
import com.crispyland.agent.memory.MemoryExtractor;
import com.crispyland.agent.memory.MemoryRouter;
import com.crispyland.agent.memory.MemoryScope;
import com.crispyland.agent.profile.Persona;
import com.crispyland.agent.memory.MemoryState;
import com.crispyland.agent.memory.Message;
import com.crispyland.agent.memory.MessageStats;
import com.crispyland.agent.memory.Summary;
import com.crispyland.agent.policy.InputPolicy;
import com.crispyland.agent.policy.OutputPolicy;
import com.crispyland.agent.task.BlockedTurn;
import com.crispyland.agent.task.PausedTurn;
import com.crispyland.agent.task.RequestShape;
import com.crispyland.agent.task.StageGate;
import com.crispyland.agent.task.TaskStage;
import com.crispyland.agent.task.TaskState;
import com.crispyland.agent.usage.ContextBudget;
import com.crispyland.agent.usage.TemplateOverhead;
import com.crispyland.agent.usage.TokenUsage;
import com.crispyland.agent.usage.TokenUsageTracker;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

/**
 * The box. One pipeline:
 * <pre>
 *   input policy -> merge config with defaults -> load history -> compress the backlog
 *       -> price the context -> LlmClient -> record token usage -> output policy -> judge
 *       -> append turn to history -> AgentResult
 * </pre>
 * The conversation is the agent's own state, not the web layer's: callers pass a
 * conversation id and the agent decides what history to send, how to window it, and when
 * to commit a turn. Knows nothing about HTTP servlets, Thymeleaf, or forms.
 */
@Service
public class Agent {

    private static final Logger log = LoggerFactory.getLogger(Agent.class);

    private final LlmClient llmClient;
    private final InputPolicy inputPolicy;
    private final OutputPolicy outputPolicy;
    private final Judge judge;
    private final TokenUsageTracker usageTracker;
    private final ConversationStore conversations;
    private final LongTermStore longTerm;
    private final InvariantStore invariants;
    private final InvariantGuard guard;
    private final ContextPlanner contextPlanner;
    private final HistoryCompressor compressor;
    private final MemoryExtractor extractor;
    private final TemplateOverhead templateOverhead;
    private final ToolBox toolBox;
    private final int maxToolRounds;
    private final AgentConfig defaults;
    private final int maxFacts;

    /**
     * Not injected: the router is a pure function of the table in {@code MemoryTag}, holds nothing,
     * and has nothing to configure. Making it a bean would suggest it could be swapped for one
     * that files things elsewhere, which is the exact property this design exists to deny.
     */
    private final MemoryRouter router = new MemoryRouter();

    /** An agent that can only talk. Every test builds this one; nothing in production does. */
    public Agent(LlmClient llmClient,
                 InputPolicy inputPolicy,
                 OutputPolicy outputPolicy,
                 Judge judge,
                 TokenUsageTracker usageTracker,
                 ConversationStore conversations,
                 LongTermStore longTerm,
                 InvariantStore invariants,
                 InvariantGuard guard,
                 ContextPlanner contextPlanner,
                 HistoryCompressor compressor,
                 MemoryExtractor extractor,
                 TemplateOverhead templateOverhead,
                 AgentProperties properties) {
        this(llmClient, inputPolicy, outputPolicy, judge, usageTracker, conversations, longTerm,
                invariants, guard, contextPlanner, compressor, extractor, templateOverhead,
                properties, ToolBox.NONE, 0);
    }

    @Autowired
    public Agent(LlmClient llmClient,
                 InputPolicy inputPolicy,
                 OutputPolicy outputPolicy,
                 Judge judge,
                 TokenUsageTracker usageTracker,
                 ConversationStore conversations,
                 LongTermStore longTerm,
                 InvariantStore invariants,
                 InvariantGuard guard,
                 ContextPlanner contextPlanner,
                 HistoryCompressor compressor,
                 MemoryExtractor extractor,
                 TemplateOverhead templateOverhead,
                 AgentProperties properties,
                 ToolBox toolBox,
                 @Value("${agent.mcp.max-tool-rounds:3}") int maxToolRounds) {
        this.toolBox = toolBox;
        this.maxToolRounds = maxToolRounds;
        this.llmClient = llmClient;
        this.inputPolicy = inputPolicy;
        this.outputPolicy = outputPolicy;
        this.judge = judge;
        this.usageTracker = usageTracker;
        this.conversations = conversations;
        this.longTerm = longTerm;
        this.invariants = invariants;
        this.guard = guard;
        this.contextPlanner = contextPlanner;
        this.compressor = compressor;
        this.extractor = extractor;
        this.templateOverhead = templateOverhead;
        this.defaults = properties.defaultConfig();
        this.maxFacts = properties.facts().maxFacts();
    }

    /**
     * Handles one turn of a continuing dialogue. Prior messages for {@code scope} are replayed to
     * the model, and the new user/assistant pair is appended on success. Any config value left
     * null falls back to the application.yml defaults.
     *
     * @throws AgentException if a policy rejects the exchange, the context window cannot
     *         hold the call, or the provider fails
     */
    public AgentResult handle(MemoryScope scope, Persona persona, String userInput, AgentConfig config) {
        return handle(scope, persona, userInput, config, PausedTurn.REFUSE);
    }

    /**
     * The same turn, with a say in what a paused task does to it.
     *
     * @param whenPaused {@link PausedTurn#ASIDE} lets a message unrelated to the task through
     *        while leaving the task frozen. It is a separate argument rather than something read
     *        off the message because only the person sending it knows whether it is an aside.
     */
    public AgentResult handle(MemoryScope scope, Persona persona, String userInput,
                              AgentConfig config, PausedTurn whenPaused) {
        return handle(scope, persona, userInput, config, whenPaused, BlockedTurn.REFUSE);
    }

    /**
     * The same turn again, with a say in what a stage the work has not reached does to it.
     *
     * @param whenBlocked what to do if the message asks for implementation while the plan is still
     *        being agreed. Separate from {@code whenPaused} rather than folded into one "overrides"
     *        argument because the two stop the turn at different points and for unrelated reasons,
     *        and a single flag would let an answer to one silently answer the other
     */
    public AgentResult handle(MemoryScope scope, Persona persona, String userInput,
                              AgentConfig config, PausedTurn whenPaused, BlockedTurn whenBlocked) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        Persona who = (persona == null) ? Persona.NONE : persona;
        String id = where.conversation();
        AgentConfig effective = effectiveConfig(who, config);

        String prompt = inputPolicy.apply(userInput);

        // Ahead of every call, including the two this turn would make before the answer: a paused
        // task refuses the turn outright rather than answering and silently declining to move.
        // Deciding per message would cost the very calls the pause exists to stop. An aside is
        // let through here and frozen further in, by the refusal in TaskState.apply.
        TaskState paused = conversations.task(id);
        boolean aside = paused.isPresent() && paused.paused();
        if (aside && whenPaused != PausedTurn.ASIDE) {
            throw new TaskPausedException(paused);
        }
        if (aside) {
            log.info("Aside on a task paused in {} — answered, but the task does not move.",
                    paused.stage().id());
        }

        // Ahead of the memory writes as well as the calls, and the order is the point: a message
        // that asks for something forbidden must not first be filed as an established fact. Left
        // until after extraction, a refused request would still teach the agent what was asked
        // for, and every later turn would answer against it.
        Invariants rules = invariants.held(where.visitor());
        InvariantGuard.Ruling ruling = guard.check(prompt, rules);
        if (ruling.breached()) {
            return refuse(where, who, effective, prompt, rules, ruling);
        }

        // Before pricing, not after: the whole point is that this turn is the one that gets
        // cheaper, and the budget the caller is shown has to be the budget that was spent.
        int compacted = compressIfDue(id);
        Read read = updateMemory(where, prompt);

        // After the extraction, because that call is where the request's shape comes from, and
        // before the answer call, because stopping the turn after paying for the answer stops
        // nothing. The gate is the other half of the state machine: TaskState.apply polices where
        // the task *says* it is, and this polices what gets done while it says so.
        StageGate.Ruling gate = StageGate.check(conversations.task(id), read.asked());
        if (gate.blocked()) {
            if (whenBlocked == BlockedTurn.REFUSE) {
                throw new StageBlockedException(conversations.task(id), gate);
            }
            if (whenBlocked == BlockedTurn.APPROVE) {
                // HUMAN, and this is the only place in the codebase that says so for this edge:
                // the button *is* the approval. The gate only ever fires on planning, so the move
                // it is standing in front of is always this one.
                applyProposedMove(id, TaskState.Proposal.toStage(TaskStage.EXECUTION),
                        TaskState.Authority.HUMAN);
            } else {
                log.info("Answered a request for execution with the task still in planning, at the "
                        + "user's word. The stage does not move, so the next one is gated again.");
            }
        }

        MemoryState memory = new MemoryState(longTerm.recall(where.visitor()),
                conversations.summary(id), conversations.facts(id), conversations.task(id),
                conversations.history(id));
        List<Message> history = memory.recent();

        // Asked now rather than at startup, because the answer changes: the switch moves, servers
        // come and go. Empty is the overwhelmingly common case and costs nothing — no `tools` key
        // is written, so the request is identical to the one this app sent before tools existed.
        //
        // Asked *before* the planner, which is the whole point of this ordering: the tools are part
        // of the prompt the window has to hold, so a budget drawn up without them is a budget for a
        // request that is not the one about to be sent.
        List<ToolSpec> offered = toolBox.available();

        // Priced before a byte leaves the process: an oversized prompt is billed as a
        // rejection, so the cheapest place to find out it will not fit is here.
        ContextPlanner.ContextPlan plan =
                contextPlanner.plan(effective, who, rules, memory, prompt, offered);
        ContextBudget budget = plan.budget();
        if (budget.trimmed()) {
            log.info("Context trim: dropped {} oldest message(s) to fit {} of {} tokens",
                    budget.droppedMessages(), budget.projectedTokens(), budget.contextWindow());
        }

        ChatRequest request = buildRequest(plan.messages(), effective).withTools(offered);

        long startedAt = System.nanoTime();
        ChatResponse response = llmClient.complete(request);
        usageTracker.record(response.usage());

        // The turn, not the call. A tool round trip is still one question from the user and one
        // answer to it, so what the page reports as this turn's cost is every call it took —
        // while the tracker keeps counting calls, because that is what the provider bills.
        TokenUsage turnUsage = response.usage();
        List<ToolRound> rounds = new ArrayList<>();
        while (response.wantsTools() && rounds.size() < maxToolRounds) {
            List<ToolResult> results = new ArrayList<>();
            for (ToolCall call : response.toolCalls()) {
                results.add(toolBox.call(call));
            }
            rounds.add(new ToolRound(response.toolCalls(), results));

            // The same request, knowing more. Everything the planner decided still holds; the only
            // difference is that the model can now see what it asked to be looked up.
            response = llmClient.complete(request.withRounds(rounds));
            turnUsage = turnUsage.plus(response.usage());
            usageTracker.record(response.usage());
        }
        long latencyMillis = (System.nanoTime() - startedAt) / 1_000_000L;

        if (response.wantsTools()) {
            // Out of rounds with the model still asking. Answering with whatever text came back is
            // wrong — there may be none — so this is a dead end worth naming rather than a silent
            // truncation of the model's plan.
            log.warn("The model was still asking for tools after {} round(s); stopping there.",
                    maxToolRounds);
        }
        if (!rounds.isEmpty()) {
            log.info("Turn used {} tool round(s): {}", rounds.size(), describe(rounds));
        }

        TokenUsage cumulative = usageTracker.cumulative();

        // The response is the only place the truth is ever stated. Feeding it back is what
        // keeps the next turn's pre-flight estimate honest — but only on a turn that is shaped
        // like the ones the estimate is for. The planner does now price the tool schemas, so
        // these turns are no longer unmodelled — they are mis-modelled in a known direction.
        // Counting the schemas as the JSON we render over-states them by roughly a quarter,
        // because the provider re-renders them into a cheaper form, so on a tool turn the
        // counted total lands *above* the reported prompt_tokens. Observing that would hand
        // TemplateOverhead a negative delta, which it clamps to zero — quietly erasing a real
        // sixty-token template cost that every toolless turn still pays. The gate stays until
        // the tool estimate is accurate enough to learn from.
        if (offered.isEmpty()) {
            templateOverhead.observe(effective.model(), budget.countedTokens(),
                    turnUsage.promptTokens());
        }

        log.info("Turn: prompt est {} / actual {} ({} history msgs, summary {} tok replacing {}), "
                        + "completion {}, window {}% used",
                budget.promptTokens(), turnUsage.promptTokens(), history.size(),
                budget.summaryTokens(), budget.replacedTokens(),
                turnUsage.completionTokens(), budget.usedPercent());

        String answer = outputPolicy.apply(response.content());
        Verdict verdict = judge.judge(prompt, answer, effective);

        // Committed only after the reply survives the output policy, so a failed turn
        // never poisons the history. Each message keeps the share of the turn it earned.
        conversations.append(id, List.of(
                Message.user(prompt).withStats(
                        MessageStats.forPrompt(turnUsage.promptTokens(), effective.model())),
                Message.assistant(answer).withStats(
                        MessageStats.forCompletion(turnUsage.completionTokens(),
                                        turnUsage.totalTokens(), latencyMillis, effective.model(),
                                        response.finishReason())
                                .withToolSteps(toolSteps(rounds)))));

        return new AgentResult(answer, effective, turnUsage, cumulative, budget,
                response.finishReason(), latencyMillis, verdict, conversations.history(id),
                compacted, InvariantGuard.Ruling.CLEAR, read.refusedMove());
    }

    /**
     * Turns a breach into the turn's answer, and commits it to the transcript like any other.
     * <p>
     * This is where invariants part company with the other refusals in here. An overflow or a
     * paused task throws, because nothing was produced and the message is best left in the box for
     * the user to edit or resend. A breach produces something worth keeping: the rule, the reason
     * it exists, and what can be done instead — which is an answer to the question, just not the
     * one that was asked for.
     * <p>
     * Writing both halves into the transcript is what stops the same refusal happening twice.
     * Dropped instead, the exchange leaves no trace the model can read, so two turns later it
     * cheerfully proposes the forbidden thing again and the user gets the same paragraph back.
     * Kept, the refusal is in the history and the next turn is answered in light of it.
     * <p>
     * Nothing is extracted from a refused message and the task does not move, both for the same
     * reason: what was asked for is not going to happen, so filing it as established fact or as
     * progress would record a thing that never took place.
     */
    private AgentResult refuse(MemoryScope where, Persona who, AgentConfig effective, String prompt,
                               Invariants rules, InvariantGuard.Ruling ruling) {
        String redirect = ruling.redirect();
        conversations.append(where.conversation(),
                List.of(Message.user(prompt), Message.assistant(redirect)));

        // The guard's own call is the only thing this turn cost, and on the forbid tier not even
        // that. Billed like any other call so a refusal is never free-looking when it was not.
        TokenUsage usage = new TokenUsage(0, 0, ruling.costTokens());
        log.info("Refused by {} — {}. The answer call was never made{}.",
                ruling.broken().label(),
                ruling.settledInJava() ? "matched on " + ruling.terms() : "ruled on by the guard",
                ruling.costTokens() == 0 ? ", and nothing was charged"
                        : ", at a cost of " + ruling.costTokens() + " token(s)");

        return new AgentResult(redirect, effective, usage, usageTracker.record(usage),
                contextPlanner.budget(effective, who, rules, memory(where)),
                "invariant", 0L, Verdict.NOT_SCORED, conversations.history(where.conversation()), 0,
                // No move to report: a refused message is never extracted from, so nothing was
                // ever proposed for the machine to turn down.
                ruling, null);
    }

    /** Read-only view of the message stack, for rendering an existing dialogue. */
    public List<Message> transcript(String conversationId) {
        return conversations.history(conversationId);
    }

    /** The notes standing in for whatever has already been compressed out of the transcript. */
    public Summary summary(String conversationId) {
        return conversations.summary(conversationId);
    }

    /** Working memory: what the task in hand has settled, as key/value. */
    public Facts facts(String conversationId) {
        return conversations.facts(conversationId);
    }

    /** Long-term memory: what is known about the visitor, whichever conversation they are in. */
    public LongTermMemory recall(String visitorId) {
        return longTerm.recall(visitorId);
    }

    /**
     * What the dialogue already costs, before anything new is typed. Lets the window be
     * watched as it fills rather than only at the turn that breaks it.
     */
    public ContextBudget budget(MemoryScope scope, Persona persona, AgentConfig config) {
        Persona who = (persona == null) ? Persona.NONE : persona;
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        // The tools are asked for here too, even though nothing is being sent. This view exists to
        // say what the next call will cost, and leaving them out would make the bar jump several
        // hundred tokens the instant Send was pressed — the estimate would be wrong in exactly the
        // moments somebody is looking at it to decide whether to trim the conversation.
        return contextPlanner.budget(effectiveConfig(who, config), who,
                invariants.held(where.visitor()), memory(where), toolBox.available());
    }

    /** The standing rules for a visitor, for rendering the page. */
    public Invariants invariants(String visitorId) {
        return invariants.held(visitorId);
    }

    /**
     * Declares a rule, or amends the one already held under the same id.
     * <p>
     * Public on the agent and reachable only from the controller — which is to say, only from a
     * form somebody filled in. Nothing on the extraction path can get here, and that is the whole
     * of what makes an invariant different from a remembered fact: a model that can mint its own
     * constraints can mint the one that permits what it wanted to do, and the mechanism becomes
     * decoration.
     */
    public Invariants declareInvariant(MemoryScope scope, Invariant invariant) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        Invariants held = invariants.declare(where.visitor(), invariant);
        log.info("Invariants rev {} — {} rule(s) now binding on this visitor.",
                held.revision(), held.binding().size());
        return held;
    }

    /** Stops a rule binding, keeping it and the reason on the record. */
    public Invariants retireInvariant(MemoryScope scope, String id, String reason) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        Invariants held = invariants.retire(where.visitor(), id, reason);
        log.info("Invariant {} retired — {}. {} rule(s) still binding.",
                id, blankAsDash(reason), held.binding().size());
        return held;
    }

    /** Puts a retired rule back in force. */
    public Invariants restoreInvariant(MemoryScope scope, String id) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        return invariants.restore(where.visitor(), id);
    }

    /**
     * What a request would actually be made with: three tiers, and the order is the whole of what
     * "enforced in code" means here.
     * <p>
     * What the caller set explicitly is someone overriding this one request on purpose and wins;
     * what the profile asks for is a standing preference and fills the gaps; the yml is what to do
     * when nobody said. Resolving against the defaults first would fill every gap before the
     * profile was consulted, leaving it nothing to apply and making it prose-only again.
     * <p>
     * Public because the page has to prefill its settings boxes with these numbers rather than the
     * raw defaults. Showing 1,024 in a box while sending 600 would be a lie, and worse, the box is
     * posted back — so the form would hand back the default as an explicit override and quietly
     * beat the profile on every single turn. That is not a display bug; it silently disables the
     * enforced half of the feature.
     */
    public AgentConfig effectiveConfig(Persona persona, AgentConfig config) {
        Persona who = (persona == null) ? Persona.NONE : persona;
        return who.applyTo((config == null) ? AgentConfig.builder().build() : config)
                .withFallback(defaults);
    }

    /** Every layer at once, gathered here so what is priced is what would have been sent. */
    private MemoryState memory(MemoryScope scope) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        String id = where.conversation();
        return new MemoryState(longTerm.recall(where.visitor()), conversations.summary(id),
                conversations.facts(id), conversations.task(id), conversations.history(id));
    }

    /**
     * Extracts what the message about to be sent establishes, routes it, and commits both halves
     * before the turn is priced — so this turn already answers against what it just taught the
     * agent, rather than the next one.
     * <p>
     * Extraction and routing are two steps on purpose. The model labels; {@link MemoryRouter}
     * decides the lifetime from a fixed table. Nothing between here and the stores consults the
     * model about where a line belongs.
     * <p>
     * Same bargain as compression: an extraction failure costs one turn of forgetfulness, never
     * the turn itself. A user asking a question does not deserve an error because a background
     * bookkeeping call timed out.
     * <p>
     * The same call also carries the task's own stage, which is why this method has stopped
     * returning just the fact block. One reading of the message serves both questions — what did
     * it establish, and did it move the job on — and splitting them into two calls would double
     * the per-turn request count to re-read text this one already has.
     */
    private Read updateMemory(MemoryScope where, String prompt) {
        String id = where.conversation();
        Facts current = conversations.facts(id);

        MemoryExtractor.Extraction extraction;
        try {
            extraction = extractor.extract(prompt,
                    keysInUse(current, longTerm.recall(where.visitor())), conversations.task(id));
        } catch (AgentException e) {
            log.warn("Memory extraction failed ({}) — continuing with the layers as they stand.",
                    e.getMessage());
            // NOTHING, so a failed call opens the gate rather than closing it. The gate rides on
            // this call; a call that did not happen has not accused anybody of anything, and
            // refusing a turn because the extractor was down would make an outage look like policy.
            return Read.NOTHING;
        }
        if (extraction.isEmpty()) {
            return Read.NOTHING;
        }

        MemoryRouter.Routed routed = router.route(extraction.lines());

        // Working memory carries the running bill for the whole extraction even when the routed
        // lines all went elsewhere, because it is the only layer counting, and a call nobody is
        // charged for is a call nobody notices the cost of.
        Facts updated = current.updatedWith(routed.working(), maxFacts, extraction.costTokens());
        if (updated.entries().equals(current.entries())) {
            updated = current.billed(extraction.costTokens());
        } else {
            log.info("Working memory rev {} — {} fact(s) held for the task in hand, {} of them agreed",
                    updated.revision(), updated.size(), updated.settled().size());
        }
        conversations.saveFacts(id, updated);

        if (!routed.longTerm().isEmpty()) {
            LongTermMemory kept = longTerm.remember(where.visitor(), routed.longTerm());
            log.info("Long-term memory rev {} — {} entry(ies) now known about this visitor",
                    kept.revision(), kept.size());
        }

        TaskState.Transition move = applyProposedMove(id, TaskState.Proposal.from(routed.stage()),
                TaskState.Authority.MODEL);

        // `asks-for` is not one of the fields a proposal is built from, precisely so that asking
        // for something cannot be the same event as the task moving to where that something
        // belongs — otherwise the gate's own input walks the task past the gate.
        return new Read(RequestShape.in(routed.stage()), move.refused() ? move : null);
    }

    /**
     * The two things the extraction call tells the turn that are not memory.
     *
     * @param asked what the message wanted done, for the gate
     * @param refusedMove the move the machine would not make, or {@code null} — which is nearly
     *                    always, and the null is the point: a field that is usually absent reads
     *                    as "nothing to report" at every call site without anyone asking it to
     */
    private record Read(RequestShape asked, TaskState.Transition refusedMove) {

        static final Read NOTHING = new Read(RequestShape.NONE, null);
    }

    /**
     * Puts a proposed move to the machine and saves it if it is allowed.
     * <p>
     * A refusal is logged at WARN and the turn carries on. That is the loudest the disagreement
     * should get: the model's idea of where the task is has drifted from the machine's, which is
     * worth knowing about and is not worth failing a user's turn over. The machine stays where it
     * was, which is the safe side of the disagreement — an unmoved task under-claims progress,
     * whereas an illegally moved one claims work that was never done.
     * <p>
     * The transition comes back rather than the state, so the caller can tell a refusal from a
     * no-op. Both leave the task exactly where it was, and only one of them is worth showing
     * anybody.
     */
    private TaskState.Transition applyProposedMove(String id, TaskState.Proposal proposal,
                                                   TaskState.Authority by) {
        TaskState current = conversations.task(id);
        TaskState.Transition transition = current.apply(proposal, by);

        if (transition.refused()) {
            log.warn("Refused a {} task transition — {}. The task stays in {}.",
                    by.name().toLowerCase(Locale.ROOT), transition.why(), current.stage().id());
            return transition;
        }
        if (!transition.moved()) {
            return transition;
        }
        conversations.saveTask(id, transition.state());
        log.info("Task rev {} — {} (step: {}, next: {} from the {})", transition.state().revision(),
                transition.why(), blankAsDash(transition.state().step()),
                blankAsDash(transition.state().next()), transition.state().awaiting().id());
        return transition;
    }

    private static String blankAsDash(String value) {
        return value.isEmpty() ? "—" : value;
    }

    /**
     * Every key the layers are already using, across both of them.
     * <p>
     * Deduplicated and flattened, because the extractor is being told what a subject is
     * <em>called</em>, not where it lives. A key held in long-term that turns up again in a task
     * message should land on the same key and be re-routed by its new tag; qualifying the list by
     * layer would instead teach the model to keep it where it was.
     */
    private static List<String> keysInUse(Facts working, LongTermMemory known) {
        LinkedHashSet<String> keys = new LinkedHashSet<>();
        working.entries().forEach(fact -> keys.add(fact.key()));
        known.entries().forEach(entry -> keys.add(entry.key()));
        return List.copyOf(keys);
    }

    /**
     * What closing the task did, or why it did not.
     *
     * @param closed   false when the state machine refused the move to {@code done}, in which case
     *                 nothing was promoted and nothing was cleared
     * @param why      the refusal, or what was promoted — the page shows this either way
     * @param longTerm the long-term block as it now stands
     */
    public record TaskClosure(boolean closed, String why, LongTermMemory longTerm) {
    }

    /**
     * Closes the task in hand: everything working memory had marked as agreed graduates to
     * long-term, the rest of the block is thrown away, and the state machine is reset.
     * <p>
     * This is the one moment a decision crosses a lifetime boundary, and it is deliberately a
     * thing the user does rather than a thing the model infers. Long-term memory is shared by
     * every branch of every conversation; promoting a decision while the task that produced it is
     * still open leaks it into the forks that exist precisely to disagree with it, and a model
     * guessing at "is this task finished?" would do exactly that on the turn it guessed wrong.
     * <p>
     * <strong>This button is the {@code → done} transition</strong>, which is what gives the
     * transition table something to actually govern. A task sitting in {@code planning} cannot be
     * closed, because nothing has been built and nothing has been checked, and the promotion would
     * write "agreed" lines about work that never happened into memory every later branch reads.
     * <p>
     * A conversation that never started a task is exempt, and that is not a loophole: the machine
     * governs tasks that have a state, and one with no state is the pre-Day-13 behaviour, where
     * closing was simply a way to file what the dialogue had agreed.
     */
    public TaskClosure finishTask(MemoryScope scope) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        String id = where.conversation();
        TaskState task = conversations.task(id);

        if (task.isPresent()) {
            TaskState.Transition closing = task.apply(
                    TaskState.Proposal.toStage(TaskStage.DONE), TaskState.Authority.HUMAN);
            if (closing.refused()) {
                log.info("Refused to close the task — {}. Nothing promoted.", closing.why());
                return new TaskClosure(false, closing.why(), longTerm.recall(where.visitor()));
            }
        }

        List<Facts.Fact> settled = conversations.facts(id).settled();
        conversations.saveFacts(id, Facts.EMPTY);
        conversations.saveTask(id, TaskState.EMPTY);

        if (settled.isEmpty()) {
            log.info("Task closed with nothing agreed — working memory cleared, long-term untouched.");
            return new TaskClosure(true, "Task closed; nothing had been agreed to keep.",
                    longTerm.recall(where.visitor()));
        }
        LongTermMemory kept = longTerm.remember(where.visitor(),
                LongTermMemory.promoted(LongTermKind.DECISION, settled));
        log.info("Task closed — promoted {} agreed fact(s) to long-term memory (rev {}) and cleared "
                + "the rest of working memory.", settled.size(), kept.revision());
        return new TaskClosure(true,
                "Task closed; %d agreed fact(s) kept.".formatted(settled.size()), kept);
    }

    /**
     * Starts a fresh task: working memory and the state machine, leaving the dialogue and the
     * visitor alone.
     * <p>
     * The other half of the boundary, and unlike closing it is not gated by the transition table.
     * Abandoning a task is not a move within the machine — it is throwing the machine away — so a
     * task stuck anywhere at all must be able to end this way. Refusing to abandon would be the
     * one refusal with no escape from it.
     */
    public void newTask(MemoryScope scope) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        conversations.saveFacts(where.conversation(), Facts.EMPTY);
        conversations.saveTask(where.conversation(), TaskState.EMPTY);
        log.info("New task — working memory and task state cleared; short-term and long-term are "
                + "untouched.");
    }

    /** Where the task in hand has got to, for rendering the page. */
    public TaskState task(String conversationId) {
        return conversations.task(conversationId);
    }

    /**
     * A person moving the task by hand.
     * <p>
     * Bound by the same transition table as the model. The button overrides <em>authority</em>,
     * not legality — if it overrode both, the table would only describe what the model does and
     * the machine would have two sets of rules, which is one more than a machine can have.
     */
    public TaskState moveTask(MemoryScope scope, TaskStage stage) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        return applyProposedMove(where.conversation(), TaskState.Proposal.toStage(stage),
                TaskState.Authority.HUMAN).state();
    }

    /**
     * Stops the machine where it stands, and with it the turns. While a task is paused a new
     * message is refused before any call is made, rather than answered with the state change
     * quietly dropped.
     * <p>
     * Pausing a conversation that never started a task does nothing. Left to
     * {@link TaskState#pause()} alone it would bump the revision and so bring a task into
     * existence — a pause that creates the thing it is pausing, and worse, one that then locks a
     * dialogue which never had a machine to begin with.
     */
    public TaskState pauseTask(MemoryScope scope) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        TaskState current = conversations.task(where.conversation());
        if (!current.isPresent()) {
            log.info("Nothing to pause — this conversation has no task.");
            return current;
        }
        TaskState paused = current.pause();
        conversations.saveTask(where.conversation(), paused);
        log.info("Task paused in {} — the state is on disk; it survives a restart.",
                paused.stage().id());
        return paused;
    }

    /** Lets the turns through again. Harmless on a task that was never paused, or never started. */
    public TaskState resumeTask(MemoryScope scope) {
        MemoryScope where = (scope == null) ? MemoryScope.of(null) : scope;
        TaskState current = conversations.task(where.conversation());
        if (!current.paused()) {
            return current;
        }
        TaskState resumed = current.resume();
        conversations.saveTask(where.conversation(), resumed);
        log.info("Task resumed in {}.", resumed.stage().id());
        return resumed;
    }

    /**
     * Folds the backlog into the summary when there is enough of it to be worth a call.
     * <p>
     * A summarization failure must not take the user's turn down with it. The worst case of
     * skipping it is a prompt that stays large for one more turn, which the overflow policy is
     * already there to catch; the worst case of propagating it is an agent that stops answering
     * because a background housekeeping call timed out.
     *
     * @return how many messages were folded, 0 if none
     */
    private int compressIfDue(String id) {
        try {
            return compressor.compact(conversations.summary(id), conversations.history(id))
                    .map(compaction -> {
                        conversations.compact(id, compaction.summary(), compaction.foldedMessages());
                        log.info("Compressed {} message(s) worth {} tokens into summary rev {} — "
                                        + "every later turn now replays notes instead",
                                compaction.foldedMessages(), compaction.foldedTokens(),
                                compaction.summary().revision());
                        return compaction.foldedMessages();
                    })
                    .orElse(0);
        } catch (AgentException e) {
            log.warn("History compression failed ({}) — continuing with the full transcript.",
                    e.getMessage());
            return 0;
        }
    }

    /** Starts a new dialogue, discarding the message stack. */
    public void reset(String conversationId) {
        conversations.clear(conversationId);
    }

    /**
     * Richer version of {@link #toolNames}: captures name + result content so the view can show
     * what each step produced and offer download links for saved files.
     */
    private static List<MessageStats.ToolStep> toolSteps(List<ToolRound> rounds) {
        List<MessageStats.ToolStep> steps = new ArrayList<>();
        for (ToolRound round : rounds) {
            for (ToolResult result : round.results()) {
                steps.add(new MessageStats.ToolStep(result.name(),
                        result.content() != null ? result.content() : "", result.failed()));
            }
        }
        return steps;
    }

    /**
     * What ran, in the order it ran, a name repeated if it ran twice.
     * <p>
     * A failed tool is named as one rather than dropped. It still cost a round trip and the model
     * still answered around it, so an answer that looks unsourced because the lookup broke should
     * say so — that is the case where the reader most needs to know.
     */
    private static List<String> toolNames(List<ToolRound> rounds) {
        List<String> names = new ArrayList<>();
        for (ToolRound round : rounds) {
            for (ToolResult result : round.results()) {
                names.add(result.failed() ? result.name() + " (failed)" : result.name());
            }
        }
        return names;
    }

    /** The same list as one line, for the log. */
    private static String describe(List<ToolRound> rounds) {
        return String.join(", ", toolNames(rounds));
    }

    private ChatRequest buildRequest(List<Message> messages, AgentConfig config) {
        return new ChatRequest(
                config.model(),
                messages,
                config.temperature(),
                config.maxCompletionTokens(),
                config.hasReasoningEffort() ? config.reasoningEffort() : null,
                config.stopSequencesOrEmpty(),
                config.hasResponseSchema() ? config.responseSchema() : null);
    }
}
