package com.crispyland.agent;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.crispyland.agent.invariant.Invariants;
import com.crispyland.agent.llm.ToolSpec;
import com.crispyland.agent.memory.Facts;
import com.crispyland.agent.profile.Limits;
import com.crispyland.agent.profile.Persona;
import com.crispyland.agent.profile.UserProfile;
import com.crispyland.agent.memory.LongTermKind;
import com.crispyland.agent.memory.LongTermMemory;
import com.crispyland.agent.memory.MemoryLayer;
import com.crispyland.agent.memory.MemoryState;
import com.crispyland.agent.memory.Message;
import com.crispyland.agent.memory.Summary;
import com.crispyland.agent.task.AwaitedFrom;
import com.crispyland.agent.task.TaskStage;
import com.crispyland.agent.task.TaskState;
import com.crispyland.agent.usage.BpeTokenCounter;
import com.crispyland.agent.usage.ContextBudget;
import com.crispyland.agent.usage.OverflowPolicy;
import com.crispyland.agent.usage.TemplateOverhead;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ContextPlannerTest {

    private static final AgentConfig CONFIG = AgentConfig.builder()
            .model("gpt-oss-120b")
            .systemPrompt("be brief")
            .maxCompletionTokens(100)
            .build();

    private static final UserProfile RUSSELL = new UserProfile("russell", "Russell", "Russian",
            "formal", new UserProfile.Format("short bullet points", 120), List.of("use emoji"),
            Limits.NONE, null);

    private static ContextPlanner planner(int window, OverflowPolicy policy) {
        return new ContextPlanner(new BpeTokenCounter(), new TemplateOverhead(), Map.of(),
                window, policy, 0.8);
    }

    @Test
    void theSegmentsAddUpToTheEstimatedPrompt() {
        ContextBudget budget = planner(1000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, MemoryState.of(history(2)), "hello").budget();

        assertThat(budget.promptTokens()).isEqualTo(budget.systemTokens() + budget.summaryTokens()
                + budget.historyTokens() + budget.inputTokens() + budget.overheadTokens());
        assertThat(budget.projectedTokens()).isEqualTo(budget.promptTokens() + 100);
        assertThat(budget.remainingTokens()).isEqualTo(1000 - budget.projectedTokens());
    }

    @Test
    void theProvidersTemplateCostIsLearnedFromTheFirstResponse() {
        TemplateOverhead overhead = new TemplateOverhead();
        ContextPlanner planner = new ContextPlanner(new BpeTokenCounter(), overhead, Map.of(),
                1000, OverflowPolicy.OFF, 0.8);

        ContextBudget first = planner.plan(CONFIG, Persona.NONE, MemoryState.EMPTY, "hello").budget();
        assertThat(first.calibrated()).isFalse();
        assertThat(first.overheadTokens()).isZero();

        // Groq reports 60 tokens more than the messages alone encode to: that gap is the
        // harmony template, and it is the same on every subsequent call.
        overhead.observe("gpt-oss-120b", first.countedTokens(), first.countedTokens() + 60);

        ContextBudget second = planner.plan(CONFIG, Persona.NONE, MemoryState.EMPTY, "hello").budget();
        assertThat(second.calibrated()).isTrue();
        assertThat(second.overheadTokens()).isEqualTo(60);
        assertThat(second.promptTokens()).isEqualTo(first.promptTokens() + 60);
    }

    @Test
    void repeatedObservationsConvergeRatherThanOscillate() {
        TemplateOverhead overhead = new TemplateOverhead();
        for (int i = 0; i < 6; i++) {
            overhead.observe("m", 100, 160);
        }

        assertThat(overhead.forModel("m")).isEqualTo(60);
        assertThat(overhead.forModel("never-seen")).isZero();
    }

    @Test
    void anOverCountIsNeverTurnedIntoADiscount() {
        TemplateOverhead overhead = new TemplateOverhead();
        // Provider billed 80 where 100 were counted: our own count was high, not the template.
        overhead.observe("m", 100, 80);

        assertThat(overhead.forModel("m")).isZero();
        assertThat(overhead.calibrated("m")).isTrue();
    }

    @Test
    void theWindowMustHoldThePromptAndTheReplyTogether() {
        // A prompt that fits on its own but leaves no room for the answer is still a failure —
        // this is the arithmetic a message-counted window cannot see.
        ContextBudget budget = planner(1000, OverflowPolicy.OFF)
                .plan(AgentConfig.builder().model("m").maxCompletionTokens(995).build(),
                        Persona.NONE, MemoryState.EMPTY, "hello").budget();

        assertThat(budget.promptTokens()).isLessThan(1000);
        assertThat(budget.overflowing()).isTrue();
    }

    @Test
    void aPerModelWindowOverridesTheDefault() {
        ContextPlanner planner = new ContextPlanner(new BpeTokenCounter(), new TemplateOverhead(),
                Map.of("gpt-oss-120b", 8192), 131_072, OverflowPolicy.OFF, 0.8);

        assertThat(planner.budget(CONFIG, Persona.NONE, MemoryState.EMPTY).contextWindow()).isEqualTo(8192);
        assertThat(planner.budget(AgentConfig.builder().model("unlisted").build(), Persona.NONE, MemoryState.EMPTY)
                .contextWindow()).isEqualTo(131_072);
    }

    @Test
    void warningFiresBeforeOverflowNotAtIt() {
        ContextPlanner planner = new ContextPlanner(new BpeTokenCounter(), new TemplateOverhead(),
                Map.of(), 130, OverflowPolicy.OFF, 0.8);

        ContextBudget budget = planner.plan(CONFIG, Persona.NONE, MemoryState.EMPTY, "hello").budget();

        assertThat(budget.overflowing()).isFalse();
        assertThat(budget.warning()).isTrue();
        assertThat(budget.status()).isEqualTo("warn");
    }

    @Test
    void failRefusesTheCallAndExplainsTheArithmetic() {
        assertThatThrownBy(() -> planner(120, OverflowPolicy.FAIL).plan(CONFIG, Persona.NONE, MemoryState.of(history(10)), "hello"))
                .isInstanceOf(ContextOverflowException.class)
                .hasMessageContaining("over by")
                .hasMessageContaining("system")
                .hasMessageContaining("history");
    }

    @Test
    void offMeasuresTheOverflowButStillBuildsTheRequest() {
        ContextPlanner.ContextPlan plan = planner(120, OverflowPolicy.OFF).plan(CONFIG, Persona.NONE, MemoryState.of(history(10)), "hello");

        assertThat(plan.budget().overflowing()).isTrue();
        assertThat(plan.budget().droppedMessages()).isZero();
        assertThat(plan.messages()).hasSize(12);
    }

    @Test
    void trimDropsOldestFirstAndKeepsTheWindowOpeningOnAUserMessage() {
        ContextPlanner.ContextPlan plan = planner(150, OverflowPolicy.TRIM).plan(CONFIG, Persona.NONE, MemoryState.of(history(10)), "hello");

        assertThat(plan.budget().trimmed()).isTrue();
        assertThat(plan.budget().overflowing()).isFalse();
        assertThat(plan.messages().get(0).role()).isEqualTo("system");
        assertThat(plan.messages().get(1).role()).isEqualTo("user");
        // The tail is kept, the head is dropped — recency is what a dialogue needs.
        assertThat(plan.messages()).extracting(Message::content).contains("assistant 9");
    }

    @Test
    void trimStillFailsWhenTheFixedPartsAloneDoNotFit() {
        // Nothing left to drop: system prompt + new message + reserved reply already overflow.
        assertThatThrownBy(() -> planner(100, OverflowPolicy.TRIM).plan(CONFIG, Persona.NONE, MemoryState.of(history(10)), "hello"))
                .isInstanceOf(ContextOverflowException.class);
    }

    @Test
    void offeredToolsArePricedAsPartOfThePromptTheyAreSentIn() {
        ContextPlanner planner = planner(10_000, OverflowPolicy.FAIL);
        ContextBudget without = planner
                .plan(CONFIG, Persona.NONE, MemoryState.of(history(2)), "hello").budget();
        ContextBudget with = planner.plan(CONFIG, Persona.NONE, Invariants.EMPTY,
                MemoryState.of(history(2)), "hello", tools()).budget();

        assertThat(without.toolTokens()).isZero();
        assertThat(without.hasTools()).isFalse();
        assertThat(with.toolTokens()).isPositive();
        // The schemas are the only difference, so they are the whole difference.
        assertThat(with.promptTokens() - without.promptTokens()).isEqualTo(with.toolTokens());
    }

    @Test
    void toolsAreInsideTheCountedTotalSoTheLearnedTemplateCostIsNotInflatedByThem() {
        // countedTokens() is what TemplateOverhead subtracts from the provider's reported
        // prompt_tokens. A segment left out of it would be attributed to the chat template and
        // would then quietly inflate every later estimate, including the toolless ones.
        ContextBudget budget = planner(10_000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, Invariants.EMPTY, MemoryState.EMPTY, "hello", tools())
                .budget();

        assertThat(budget.countedTokens()).isEqualTo(budget.systemTokens() + budget.inputTokens()
                + budget.toolTokens());
    }

    @Test
    void aToolListLargeEnoughToOverflowTheWindowIsRefusedBeforeTheCallIsPaidFor() {
        // The defect this closes: priced at zero, these schemas sailed through the check and the
        // provider rejected the call — which is billed.
        assertThatThrownBy(() -> planner(200, OverflowPolicy.FAIL).plan(CONFIG, Persona.NONE,
                Invariants.EMPTY, MemoryState.EMPTY, "hello", tools()))
                .isInstanceOf(ContextOverflowException.class);
    }

    @Test
    void trimSpendsHistoryToMakeRoomForToolsBecauseHalfAToolListIsNotACheaperRequest() {
        MemoryState memory = MemoryState.of(history(10));
        // A window with room for the schemas and half the history. Derived rather than guessed:
        // a literal would silently stop testing anything the day the system prompt, the tool
        // schemas or the tokenizer changed underneath it.
        ContextBudget roomy = planner(10_000, OverflowPolicy.TRIM).plan(CONFIG, Persona.NONE,
                Invariants.EMPTY, memory, "hello", tools()).budget();
        int window = (int) (roomy.projectedTokens() - roomy.historyTokens() / 2);
        ContextPlanner planner = planner(window, OverflowPolicy.TRIM);

        ContextPlanner.ContextPlan without =
                planner.plan(CONFIG, Persona.NONE, memory, "hello");
        ContextPlanner.ContextPlan with = planner.plan(CONFIG, Persona.NONE, Invariants.EMPTY,
                memory, "hello", tools());

        assertThat(without.budget().trimmed()).isFalse();
        // Tools cannot be half-offered, so history is what pays for them.
        assertThat(with.budget().droppedMessages())
                .isGreaterThan(without.budget().droppedMessages());
        assertThat(with.budget().toolTokens()).isPositive();
        assertThat(with.budget().overflowing()).isFalse();
    }

    /** Two tools shaped the way a server actually sends them — nested schema, mixed types. */
    private static List<ToolSpec> tools() {
        return List.of(
                new ToolSpec("getWeather", "Get the current weather for a city.",
                        Map.of("type", "object",
                                "properties", Map.of("city", Map.of("type", "string",
                                        "description", "City name, optionally with a country.")),
                                "required", List.of("city"))),
                new ToolSpec("convert", "Convert an amount from one currency to another.",
                        Map.of("type", "object",
                                "properties", Map.of(
                                        "amount", Map.of("type", "number",
                                                "description", "Amount in the source currency."),
                                        "from", Map.of("type", "string",
                                                "description", "ISO 4217 source currency.")),
                                "required", List.of("amount", "from"))));
    }

    @Test
    void anEmptyDialogueStillCostsTheSystemPromptAndTheReservedReply() {
        ContextBudget budget = planner(1000, OverflowPolicy.FAIL).budget(CONFIG, Persona.NONE, MemoryState.EMPTY);

        assertThat(budget.historyTokens()).isZero();
        assertThat(budget.inputTokens()).isZero();
        assertThat(budget.systemTokens()).isPositive();
        assertThat(budget.reservedCompletionTokens()).isEqualTo(100);
    }

    @Test
    void theSummaryIsSentAsItsOwnSegmentAheadOfTheRetainedTurns() {
        Summary summary = Summary.EMPTY.rewrittenAs("user is called Nur", 8, 400, 60);

        ContextPlanner.ContextPlan plan = planner(1000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, MemoryState.of(summary, history(2)), "hello");

        // System prompt first, then what was forgotten, then what is still remembered verbatim.
        assertThat(plan.messages()).extracting(Message::role)
                .containsExactly("system", "system", "user", "assistant", "user");
        assertThat(plan.messages().get(1).content()).contains("user is called Nur");
        assertThat(plan.budget().summaryTokens()).isPositive();
        // The saving is measured against what those 400 tokens of messages used to cost.
        assertThat(plan.budget().replacedTokens()).isEqualTo(400);
        assertThat(plan.budget().savedTokens())
                .isEqualTo(400 - plan.budget().summaryTokens());
        assertThat(plan.budget().uncompressedPromptTokens())
                .isEqualTo(plan.budget().promptTokens() + plan.budget().savedTokens());
    }

    @Test
    void withoutASummaryNothingAboutTheBudgetChanges() {
        ContextBudget budget = planner(1000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, MemoryState.of(history(2)), "hello").budget();

        assertThat(budget.compressed()).isFalse();
        assertThat(budget.summaryTokens()).isZero();
        assertThat(budget.savedTokens()).isZero();
        assertThat(budget.uncompressedPromptTokens()).isEqualTo(budget.promptTokens());
    }

    @Test
    void theRetainedTranscriptIsReplayedWholeBecauseNothingIsCutAtReadTime() {
        // There used to be a second, read-time window here on top of the compressor's. Two
        // boundaries meant a message could be outside the window yet not yet folded — remembered
        // by neither, and nothing said so. Short-term memory is now bounded only where it is
        // written, so whatever the store still holds is exactly what the model sees.
        ContextPlanner.ContextPlan plan = planner(1000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, MemoryState.of(history(10)), "hello");

        assertThat(plan.messages()).hasSize(12);
        assertThat(plan.messages()).extracting(Message::content)
                .containsSequence("user 0", "assistant 1")
                .containsSequence("user 8", "assistant 9");
        assertThat(plan.budget().trimmed()).isFalse();
    }

    @Test
    void workingMemoryIsSentAheadOfTheTranscriptAndPricedOnItsOwn() {
        Facts working = Facts.EMPTY.updatedWith(List.of(
                new Facts.Fact("database", "Postgres 16"),
                new Facts.Fact("deadline", "end of Q3")), 12, 90);

        ContextPlanner.ContextPlan plan = planner(1000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, MemoryState.of(working, history(4)), "hello");

        assertThat(plan.messages()).extracting(Message::role)
                .containsExactly("system", "system", "user", "assistant", "user", "assistant", "user");
        assertThat(plan.messages().get(1).content()).contains("database: Postgres 16");
        assertThat(plan.budget().workingTokens()).isPositive();
        assertThat(plan.budget().summaryTokens()).isZero();
        assertThat(plan.budget().promptTokens()).isEqualTo(plan.budget().systemTokens()
                + plan.budget().workingTokens() + plan.budget().historyTokens()
                + plan.budget().inputTokens() + plan.budget().overheadTokens());
    }

    @Test
    void everyLayerPresentIsSentAndPricedSeparately() {
        // The layers are not alternatives. Holding all three means all three are in the prompt,
        // each with its own line in the budget — which is what makes "where did that answer come
        // from?" a question with an answer.
        MemoryState all = new MemoryState(
                LongTermMemory.EMPTY.updatedWith(List.of(
                        new LongTermMemory.Entry(LongTermKind.PROFILE, "name", "Nur")), 24),
                Summary.EMPTY.rewrittenAs("they argued about databases", 8, 400, 60),
                Facts.EMPTY.updatedWith(List.of(new Facts.Fact("database", "Postgres 16")), 12, 90),
                TaskState.EMPTY,
                history(2));

        ContextPlanner.ContextPlan plan = planner(2000, OverflowPolicy.FAIL).plan(CONFIG, Persona.NONE, all, "hello");
        ContextBudget budget = plan.budget();

        // Most durable first, and working memory before the notes: the block outranks the
        // transcript it summarizes, and both outrank a profile written weeks ago.
        assertThat(plan.messages().get(1).content()).contains("name: Nur");
        assertThat(plan.messages().get(2).content()).contains("database: Postgres 16");
        assertThat(plan.messages().get(3).content()).contains("they argued about databases");
        assertThat(budget.longTermTokens()).isPositive();
        assertThat(budget.workingTokens()).isPositive();
        assertThat(budget.summaryTokens()).isPositive();
        assertThat(budget.replacedTokens()).isEqualTo(400);
        assertThat(budget.memoryTokens()).isEqualTo(budget.longTermTokens()
                + budget.workingTokens() + budget.summaryTokens() + budget.historyTokens());
        // The breakdown accounts for the whole prompt. A total that does not match its own parts
        // sends the reader looking for the saving in the wrong layer.
        assertThat(budget.promptTokens()).isEqualTo(budget.systemTokens()
                + budget.memoryTokens() + budget.inputTokens() + budget.overheadTokens());
    }

    @Test
    void theTaskBlockIsSentLastOfTheBlocksBecauseItIsTheNewestThingTheModelIsTold() {
        // Position is the argument. Everything above it is background or record; this is where the
        // job actually is, so on a conflict it has to be the block the model read most recently.
        // It also sits above the replayed tail, which stops the last six messages from reading as
        // a more current account of the work than the state that was maintained deliberately.
        MemoryState all = new MemoryState(
                LongTermMemory.EMPTY.updatedWith(List.of(
                        new LongTermMemory.Entry(LongTermKind.PROFILE, "name", "Nur")), 24),
                Summary.EMPTY.rewrittenAs("they argued about databases", 8, 400, 60),
                Facts.EMPTY.updatedWith(List.of(new Facts.Fact("database", "Postgres 16")), 12, 90),
                new TaskState(TaskStage.EXECUTION, "writing the migration", "review it",
                        AwaitedFrom.USER, false, 2),
                history(2));

        ContextPlanner.ContextPlan plan = planner(2000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, all, "hello");

        assertThat(plan.messages().get(4).content())
                .contains("stage: execution")
                .contains("writing the migration")
                // The two ways a resumed conversation wastes a turn, forbidden by name.
                .contains("do not ask what you")
                .contains("do not start over");
        assertThat(plan.messages().get(5).role()).isEqualTo("user");
    }

    @Test
    void theTaskBlockIsPricedAsMemoryBecauseTheAgentInferredIt() {
        // Unlike the profile, which the user wrote and therefore is not the agent's to economise
        // on, the task state is something the agent worked out — so it belongs inside the number
        // that answers "is what I am remembering worth what it costs".
        MemoryState withTask = MemoryState.of(new TaskState(TaskStage.VALIDATION,
                "checking the migration", "confirm the row counts", AwaitedFrom.USER, false, 4),
                history(2));

        ContextBudget budget = planner(2000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, withTask, "hello").budget();

        assertThat(budget.taskTokens()).isPositive();
        assertThat(budget.hasTask()).isTrue();
        assertThat(budget.memoryTokens())
                .isEqualTo(budget.longTermTokens() + budget.workingTokens() + budget.taskTokens()
                        + budget.summaryTokens() + budget.historyTokens());
        assertThat(budget.promptTokens()).isEqualTo(budget.systemTokens()
                + budget.memoryTokens() + budget.inputTokens() + budget.overheadTokens());
    }

    @Test
    void aConversationWithNoTaskSendsNoTaskBlockAndIsChargedNothingForIt() {
        ContextPlanner.ContextPlan plan = planner(2000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, MemoryState.of(TaskState.EMPTY, List.of()), "hello");

        assertThat(plan.messages()).extracting(Message::role).containsExactly("system", "user");
        assertThat(plan.budget().taskTokens()).isZero();
        assertThat(plan.budget().hasTask()).isFalse();
    }

    @Test
    void longTermMemorySurvivesTheResetThatEmptiesTheOtherTwo() {
        LongTermMemory known = LongTermMemory.EMPTY.updatedWith(List.of(
                new LongTermMemory.Entry(LongTermKind.PROFILE, "name", "Nur"),
                new LongTermMemory.Entry(LongTermKind.DECISION, "store", "Postgres 16")), 24);

        // A reset leaves the visitor's layer intact and everything conversation-scoped empty —
        // this is the state the very next prompt is built from.
        ContextPlanner.ContextPlan plan = planner(2000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, MemoryState.of(known, List.of()), "hello");

        assertThat(plan.messages()).extracting(Message::role)
                .containsExactly("system", "system", "user");
        // Grouped under headings, because the heading is what says how binding a line is:
        // a decision must be built on, a profile line is background.
        assertThat(plan.messages().get(1).content())
                .contains("Profile (Who you are):")
                .contains("Decisions (What was agreed):");
        assertThat(plan.budget().historyTokens()).isZero();
        assertThat(plan.budget().workingTokens()).isZero();
        assertThat(plan.budget().longTermTokens()).isPositive();
    }

    @Test
    void theProfileIsSentAboveEveryMemoryLayerBecauseItOutranksThem() {
        // The layers are ordered oldest-first. The profile breaks that on purpose: it is placed by
        // authority, not by age, so the model reads "they asked for bullet points" before it reads
        // anything it worked out about them. Sent second, directly under the system prompt.
        MemoryState all = new MemoryState(
                LongTermMemory.EMPTY.updatedWith(List.of(
                        new LongTermMemory.Entry(LongTermKind.PROFILE, "style", "long essays")), 24),
                Summary.EMPTY.rewrittenAs("they argued about databases", 8, 400, 60),
                Facts.EMPTY.updatedWith(List.of(new Facts.Fact("database", "Postgres 16")), 12, 90),
                TaskState.EMPTY,
                history(2));

        ContextPlanner.ContextPlan plan = planner(2000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.of(RUSSELL), all, "hello");

        assertThat(plan.messages().get(0).content()).isEqualTo("be brief");
        assertThat(plan.messages().get(1).content()).contains("Russell").contains("this wins");
        assertThat(plan.messages().get(2).content()).contains("style: long essays");
        assertThat(plan.messages().get(3).content()).contains("database: Postgres 16");
    }

    @Test
    void theProfileIsPricedOnItsOwnLineAndNotCountedAsMemory() {
        ContextBudget budget = planner(2000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.of(RUSSELL), MemoryState.of(history(2)), "hello").budget();

        // In the prompt total, because it is real tokens on every call...
        assertThat(budget.profileTokens()).isPositive();
        assertThat(budget.hasProfile()).isTrue();
        assertThat(budget.promptTokens()).isEqualTo(budget.systemTokens() + budget.profileTokens()
                + budget.memoryTokens() + budget.inputTokens() + budget.overheadTokens());
        // ...but not in memoryTokens, which exists to answer "is this layer worth what it costs?".
        // A profile is not up for that question: the user asked for it.
        assertThat(budget.memoryTokens()).isEqualTo(budget.historyTokens());
    }

    @Test
    void noProfileCostsNothingAtAll() {
        ContextPlanner.ContextPlan plan = planner(2000, OverflowPolicy.FAIL)
                .plan(CONFIG, Persona.NONE, MemoryState.of(history(2)), "hello");

        assertThat(plan.budget().profileTokens()).isZero();
        assertThat(plan.budget().hasProfile()).isFalse();
        // No empty system message standing in for the block it does not have.
        assertThat(plan.messages()).extracting(Message::role)
                .containsExactly("system", "user", "assistant", "user");
    }

    @Test
    void theLayersAreSeparatedByScopeAndLifetimeNotByWhatTheyHold() {
        assertThat(MemoryLayer.SHORT_TERM.id()).isEqualTo("short-term");
        assertThat(MemoryLayer.LONG_TERM.id()).isEqualTo("long-term");

        // No two layers share both answers — that is what stops them collapsing into one store.
        assertThat(MemoryLayer.values()).extracting(MemoryLayer::scope).doesNotHaveDuplicates();
        assertThat(MemoryLayer.values()).extracting(MemoryLayer::lifetime).doesNotHaveDuplicates();
    }

    /** {@code count} messages alternating user/assistant, as a real transcript would be. */
    private static List<Message> history(int count) {
        List<Message> messages = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            messages.add(i % 2 == 0 ? Message.user("user " + i) : Message.assistant("assistant " + i));
        }
        return messages;
    }
}
