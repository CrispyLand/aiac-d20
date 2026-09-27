package com.crispyland.agent.memory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.tuple;

import com.crispyland.agent.task.AwaitedFrom;
import com.crispyland.agent.task.TaskStage;
import com.crispyland.agent.task.TaskState;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/** Restarting the JVM is simulated by building a second store over the same file. */
class JsonFileConversationStoreTest {

    @TempDir
    Path directory;

    private Path file;

    @BeforeEach
    void setUp() {
        file = directory.resolve("nested").resolve("conversations.json");
    }

    private JsonFileConversationStore store(int maxMessages) {
        return new JsonFileConversationStore(JsonMapper.builder().build(), file, maxMessages);
    }

    @Test
    void aDialogueSurvivesTheProcessThatWroteIt() {
        store(20).append("c1", List.of(Message.user("my name is Nur"), Message.assistant("Hello, Nur.")));

        // Fresh instance, same file — this is the restart.
        assertThat(store(20).history("c1"))
                .extracting(Message::role, Message::content)
                .containsExactly(tuple("user", "my name is Nur"), tuple("assistant", "Hello, Nur."));
    }

    @Test
    void conversationsStayIsolatedAcrossARestart() {
        JsonFileConversationStore first = store(20);
        first.append("alice", List.of(Message.user("hello from alice")));
        first.append("bob", List.of(Message.user("hello from bob")));

        JsonFileConversationStore restarted = store(20);
        assertThat(restarted.history("alice")).extracting(Message::content).containsExactly("hello from alice");
        assertThat(restarted.history("bob")).extracting(Message::content).containsExactly("hello from bob");
    }

    @Test
    void perMessageStatsRoundTripSoTheRestoredTranscriptStillRenders() {
        store(20).append("c1", List.of(
                Message.user("hello").withStats(MessageStats.forPrompt(10, "gpt-oss-20b")),
                Message.assistant("hi").withStats(
                        MessageStats.forCompletion(5, 15, 380, "gpt-oss-20b", "stop"))));

        List<Message> restored = store(20).history("c1");
        assertThat(restored.get(0).stats().promptTokens()).isEqualTo(10);
        assertThat(restored.get(1).stats()).isEqualTo(
                new MessageStats(0, 5, 15, 380, "gpt-oss-20b", "stop", List.of(), List.of()));
    }

    /**
     * The badge on a restored message is only as good as this: the names are the one trace of tool
     * use that survives the turn, so losing them on reload would leave a transcript claiming every
     * answer was written from memory alone. Order and repeats are asserted because both are the
     * content — two lookups are not one.
     */
    @Test
    void toolsUsedSurviveTheRestartSoAnOldAnswerStillSaysWhatItLookedUp() {
        store(20).append("c1", List.of(Message.assistant("two events, one task").withStats(
                MessageStats.forCompletion(5, 15, 380, "gpt-oss-20b", "stop")
                        .withToolsUsed(List.of("getSchedule", "getTasks", "getSchedule")))));

        assertThat(store(20).history("c1").get(0).stats().toolsUsed())
                .containsExactly("getSchedule", "getTasks", "getSchedule");
    }

    /** A transcript written before the field existed still loads, reading as "answered alone". */
    @Test
    void aMessageWithNoRecordedToolsReadsAsHavingUsedNone() {
        store(20).append("c1", List.of(Message.assistant("hi").withStats(
                MessageStats.forCompletion(5, 15, 380, "gpt-oss-20b", "stop"))));

        assertThat(store(20).history("c1").get(0).stats().toolsUsed()).isEmpty();
    }

    @Test
    void theWindowIsAppliedBeforeAnythingIsWritten() {
        JsonFileConversationStore windowed = store(2);
        windowed.append("c1", List.of(Message.user("first"), Message.assistant("reply 1")));
        windowed.append("c1", List.of(Message.user("second"), Message.assistant("reply 2")));

        // The trimmed stack is what persists — dropped turns are not hiding in the file.
        assertThat(store(2).history("c1"))
                .extracting(Message::role, Message::content)
                .containsExactly(tuple("user", "second"), tuple("assistant", "reply 2"));
    }

    @Test
    void aCompactedConversationRestoresWithItsSummaryInsteadOfItsMessages() {
        JsonFileConversationStore first = store(20);
        first.append("c1", List.of(Message.user("my name is Nur"), Message.assistant("Hello, Nur."),
                Message.user("still here"), Message.assistant("Yes.")));
        first.compact("c1", Summary.EMPTY.rewrittenAs("user is called Nur", 2, 40, 60), 2);

        JsonFileConversationStore restarted = store(20);
        // The folded messages are gone from disk; the notes that replaced them are not.
        assertThat(restarted.history("c1")).extracting(Message::content)
                .containsExactly("still here", "Yes.");
        assertThat(restarted.summary("c1"))
                .isEqualTo(new Summary("user is called Nur", 1, 2, 40, 60));
    }

    @Test
    void anUncompressedConversationRestoresWithNoSummary() {
        store(20).append("c1", List.of(Message.user("hello")));

        assertThat(store(20).summary("c1")).isEqualTo(Summary.EMPTY);
    }

    @Test
    void clearingRemovesTheConversationFromDisk() {
        JsonFileConversationStore first = store(20);
        first.append("c1", List.of(Message.user("hello")));
        first.clear("c1");

        assertThat(store(20).history("c1")).isEmpty();
    }

    @Test
    void aCorruptFileStartsEmptyInsteadOfBrickingStartup() throws IOException {
        Files.createDirectories(file.getParent());
        Files.writeString(file, "{{{ not json");

        assertThatCode(() -> assertThat(store(20).history("c1")).isEmpty()).doesNotThrowAnyException();
    }

    @Test
    void aFileWrittenBeforeFactsHadASettledFlagStillLoads() throws IOException {
        // Adding one field to the format must never be able to erase a history that predates it.
        // Absent reads as false, which is the safe direction: an unmarked line is scratch, so the
        // worst case is that it is dropped at the task boundary rather than promoted to permanent
        // memory on nobody's authority.
        Files.createDirectories(file.getParent());
        Files.writeString(file, """
                {"conversations":{"c1":{
                  "facts":{"revision":1,"buildTokens":60,
                           "entries":[{"key":"database","value":"Postgres 16"}]},
                  "messages":[{"role":"user","content":"hello"}]}}}""");

        JsonFileConversationStore restored = store(20);

        assertThat(restored.facts("c1").entries())
                .containsExactly(new Facts.Fact("database", "Postgres 16", false));
        assertThat(restored.history("c1")).hasSize(1);
    }

    @Test
    void aPausedTaskSurvivesTheRestartThatThePauseWasFor() {
        // This is the day's requirement in one test: stop mid-job, kill the process, come back and
        // still be in execution on the same step. A task state that lived only in memory would
        // make "pause" mean "abandon", which is the opposite of what the button claims.
        JsonFileConversationStore first = store(20);
        first.append("c1", List.of(Message.user("let's do the migration")));
        first.saveTask("c1", new TaskState(TaskStage.EXECUTION, "writing the migration script",
                "review the script", AwaitedFrom.USER, true, 3));

        assertThat(store(20).task("c1")).isEqualTo(new TaskState(TaskStage.EXECUTION,
                "writing the migration script", "review the script", AwaitedFrom.USER, true, 3));
    }

    @Test
    void theStageIsWrittenByNameSoReorderingTheEnumCannotRewriteHistory() throws IOException {
        // An ordinal would mean that inserting a stage between planning and execution silently
        // moves every stored task one step along. The name is the only stable identity an enum has.
        JsonFileConversationStore first = store(20);
        first.append("c1", List.of(Message.user("hello")));
        first.saveTask("c1", new TaskState(TaskStage.VALIDATION, "checking", "",
                AwaitedFrom.AGENT, false, 1));

        JsonNode task = JsonMapper.builder().build().readTree(Files.readString(file))
                .path("conversations").path("c1").path("task");

        assertThat(task.path("stage").stringValue()).isEqualTo("validation");
        assertThat(task.path("waiting").stringValue()).isEqualTo("agent");
    }

    @Test
    void aConversationThatNeverStartedATaskRestoresWithNoTaskAtAll() {
        // isPresent() is what the page and the prompt both branch on, so an empty task must not
        // come back looking like a task that is sitting in planning.
        store(20).append("c1", List.of(Message.user("hello")));

        assertThat(store(20).task("c1")).isEqualTo(TaskState.EMPTY);
        assertThat(store(20).task("c1").isPresent()).isFalse();
    }

    @Test
    void aFileWrittenBeforeTasksExistedStillLoads() throws IOException {
        Files.createDirectories(file.getParent());
        Files.writeString(file, """
                {"conversations":{"c1":{"messages":[{"role":"user","content":"hello"}]}}}""");

        assertThat(store(20).task("c1")).isEqualTo(TaskState.EMPTY);
        assertThat(store(20).history("c1")).hasSize(1);
    }

    @Test
    void aFullLengthForkInheritsTheTaskButAMidpointForkStartsClean() {
        // Same rule as the facts, and for the same reason. A stage is not message-addressable:
        // rewinding to message three cannot say which stage the job was in at the time, so a
        // midpoint fork that inherited the stage would claim progress it has no record of.
        JsonFileConversationStore first = store(20);
        first.append("c1", List.of(Message.user("one"), Message.assistant("two"),
                Message.user("three"), Message.assistant("four")));
        first.saveTask("c1", new TaskState(TaskStage.EXECUTION, "step", "next",
                AwaitedFrom.USER, false, 2));

        first.copy("c1", "whole", 4);
        first.copy("c1", "rewound", 2);

        assertThat(first.task("whole").stage()).isEqualTo(TaskStage.EXECUTION);
        assertThat(first.task("rewound")).isEqualTo(TaskState.EMPTY);
    }

    @Test
    void clearingRemovesTheTaskAlongWithTheMessages() {
        JsonFileConversationStore first = store(20);
        first.append("c1", List.of(Message.user("hello")));
        first.saveTask("c1", new TaskState(TaskStage.EXECUTION, "step", "", AwaitedFrom.USER, false, 1));
        first.clear("c1");

        assertThat(store(20).task("c1")).isEqualTo(TaskState.EMPTY);
    }

    @Test
    void aMissingFileIsSimplyAnEmptyDialogue() {
        assertThat(store(20).history("c1")).isEmpty();
        assertThat(file).doesNotExist();
    }

    @Test
    void writesLeaveNoTemporaryFileBehind() {
        store(20).append("c1", List.of(Message.user("hello")));

        assertThat(file).exists();
        assertThat(file.resolveSibling("conversations.json.tmp")).doesNotExist();
    }
}
