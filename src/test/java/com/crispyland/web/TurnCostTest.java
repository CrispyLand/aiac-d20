package com.crispyland.web;

import static org.assertj.core.api.Assertions.assertThat;

import com.crispyland.agent.memory.Message;
import com.crispyland.agent.memory.MessageStats;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** The growth table is a view, so the only thing to prove is that it reads the stack correctly. */
class TurnCostTest {

    private static final String MODEL = "gpt-oss-20b";

    @Test
    @DisplayName("an empty dialogue has no cost to show")
    void anEmptyDialogueHasNoCostToShow() {
        assertThat(TurnCost.series(null)).isEmpty();
        assertThat(TurnCost.series(List.of())).isEmpty();
    }

    @Test
    @DisplayName("a turn pairs the prompt that was paid with the reply it bought")
    void aTurnPairsThePromptThatWasPaidWithTheReplyItBought() {
        List<Message> transcript = new ArrayList<>();
        addTurn(transcript, 100, 20, 120, 900, "stop");

        List<TurnCost> turns = TurnCost.series(transcript);

        assertThat(turns).hasSize(1);
        TurnCost only = turns.getFirst();
        assertThat(only.turn()).isEqualTo(1);
        assertThat(only.promptTokens()).isEqualTo(100);
        assertThat(only.completionTokens()).isEqualTo(20);
        assertThat(only.totalTokens()).isEqualTo(120);
        assertThat(only.latencyMillis()).isEqualTo(900);
        assertThat(only.model()).isEqualTo(MODEL);
    }

    @Test
    @DisplayName("the running total accumulates across turns")
    void theRunningTotalAccumulatesAcrossTurns() {
        List<Message> transcript = new ArrayList<>();
        addTurn(transcript, 100, 20, 120, 900, "stop");
        addTurn(transcript, 240, 30, 270, 800, "stop");
        addTurn(transcript, 390, 25, 415, 850, "stop");

        List<TurnCost> turns = TurnCost.series(transcript);

        assertThat(turns).extracting(TurnCost::turn).containsExactly(1, 2, 3);
        assertThat(turns).extracting(TurnCost::runningTotal).containsExactly(120L, 390L, 805L);
    }

    @Test
    @DisplayName("the prompt climbs even when the messages do not — the point of the table")
    void thePromptClimbsEvenWhenTheMessagesDoNot() {
        List<Message> transcript = new ArrayList<>();
        addTurn(transcript, 100, 20, 120, 900, "stop");
        addTurn(transcript, 240, 30, 270, 800, "stop");
        addTurn(transcript, 390, 25, 415, 850, "stop");

        List<TurnCost> turns = TurnCost.series(transcript);

        assertThat(turns).extracting(TurnCost::promptTokens).isSorted();
        // Every turn re-sends everything before it, so the prompt grows faster than the replies do.
        assertThat(turns.getLast().promptTokens() - turns.getFirst().promptTokens())
                .isGreaterThan(turns.getLast().completionTokens());
    }

    @Test
    @DisplayName("bars are scaled against the largest prompt, and the largest fills the row")
    void barsAreScaledAgainstTheLargestPrompt() {
        List<Message> transcript = new ArrayList<>();
        addTurn(transcript, 100, 20, 120, 900, "stop");
        addTurn(transcript, 400, 20, 420, 900, "stop");

        List<TurnCost> turns = TurnCost.series(transcript);

        assertThat(turns.get(0).barPercent()).isEqualTo(25);
        assertThat(turns.get(1).barPercent()).isEqualTo(100);
    }

    @Test
    @DisplayName("a turn too small to see still gets a sliver of a bar")
    void aTurnTooSmallToSeeStillGetsASliverOfABar() {
        List<Message> transcript = new ArrayList<>();
        addTurn(transcript, 1, 5, 6, 100, "stop");
        addTurn(transcript, 10_000, 5, 10_005, 100, "stop");

        assertThat(TurnCost.series(transcript).getFirst().barPercent()).isEqualTo(2);
    }

    @Test
    @DisplayName("a reply cut off at the token limit is flagged in the row that paid for it")
    void aReplyCutOffAtTheTokenLimitIsFlagged() {
        List<Message> transcript = new ArrayList<>();
        addTurn(transcript, 100, 20, 120, 900, "stop");
        addTurn(transcript, 240, 16, 256, 800, "length");

        List<TurnCost> turns = TurnCost.series(transcript);

        assertThat(turns.get(0).truncated()).isFalse();
        assertThat(turns.get(1).truncated()).isTrue();
    }

    @Test
    @DisplayName("the system prompt has no stats and is not a turn")
    void theSystemPromptIsNotATurn() {
        List<Message> transcript = new ArrayList<>();
        transcript.add(Message.system("You are helpful."));
        addTurn(transcript, 100, 20, 120, 900, "stop");

        assertThat(TurnCost.series(transcript)).hasSize(1);
    }

    @Test
    @DisplayName("a history trimmed from the front still reads cleanly, starting mid-dialogue")
    void aTrimmedHistoryStillReadsCleanly() {
        // The message window dropped the first user message, so the stack opens on an assistant
        // reply whose prompt is gone. It must still be counted, not silently paired with the
        // next turn's prompt.
        List<Message> transcript = new ArrayList<>();
        transcript.add(Message.assistant("…the tail of an older answer.")
                .withStats(MessageStats.forCompletion(30, 300, 700, MODEL, "stop")));
        addTurn(transcript, 500, 20, 520, 900, "stop");

        List<TurnCost> turns = TurnCost.series(transcript);

        assertThat(turns).hasSize(2);
        assertThat(turns.getFirst().promptTokens()).isZero();
        assertThat(turns.getFirst().totalTokens()).isEqualTo(300);
        assertThat(turns.getLast().promptTokens()).isEqualTo(500);
        assertThat(turns.getLast().runningTotal()).isEqualTo(820);
    }

    @Test
    @DisplayName("a question still waiting on its answer is not yet a turn")
    void aQuestionStillWaitingOnItsAnswerIsNotYetATurn() {
        List<Message> transcript = new ArrayList<>();
        addTurn(transcript, 100, 20, 120, 900, "stop");
        transcript.add(Message.user("and then?")
                .withStats(MessageStats.forPrompt(240, MODEL)));

        assertThat(TurnCost.series(transcript)).hasSize(1);
    }

    private static void addTurn(List<Message> transcript, long prompt, long completion,
                                long total, long latency, String finishReason) {
        transcript.add(Message.user("q").withStats(MessageStats.forPrompt(prompt, MODEL)));
        transcript.add(Message.assistant("a").withStats(
                MessageStats.forCompletion(completion, total, latency, MODEL, finishReason)));
    }
}
