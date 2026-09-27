package com.crispyland.mcpserver.notes;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NotesToolsTest {

    @TempDir
    Path tempDir;

    private NotesTools tools(NotesSummarizer summarizer) {
        NotesProperties props = new NotesProperties(tempDir.toString(), null);
        return new NotesTools(summarizer, props);
    }

    // ── saveToFile ────────────────────────────────────────────────────────────

    @Test
    void saveToFileWritesContentAndReturnsAbsolutePath() throws IOException {
        NotesTools tools = tools(NotesSummarizer.NONE);
        String result = tools.saveToFile("# Notes\nHello world", "2026-09-27.md");

        assertThat(result).startsWith("Saved to");
        Path saved = tempDir.resolve("2026-09-27.md");
        assertThat(Files.readString(saved)).isEqualTo("# Notes\nHello world");
    }

    @Test
    void saveToFileCreatesTheDirectoryIfItDoesNotExist() throws IOException {
        Path subDir = tempDir.resolve("sub/dir");
        NotesProperties props = new NotesProperties(subDir.toString(), null);
        NotesTools tools = new NotesTools(NotesSummarizer.NONE, props);

        tools.saveToFile("content", "note.md");

        assertThat(Files.exists(subDir.resolve("note.md"))).isTrue();
    }

    @Test
    void pathSeparatorsInFilenameAreStrippedRatherThanEscaping() throws IOException {
        NotesTools tools = tools(NotesSummarizer.NONE);
        String result = tools.saveToFile("content", "../../etc/passwd");

        // The dangerous path segments are stripped; the file lands in the notes dir, not /etc.
        assertThat(result).startsWith("Saved to");
        assertThat(result).doesNotContain("/etc/passwd");
        assertThat(Files.exists(tempDir.resolve("etcpasswd"))).isTrue();
    }

    @Test
    void aBlankContentReturnsAnErrorAndWritesNothing() {
        NotesTools tools = tools(NotesSummarizer.NONE);
        String result = tools.saveToFile("  ", "note.md");

        assertThat(result).contains("No content");
        assertThat(tempDir.resolve("note.md").toFile().exists()).isFalse();
    }

    @Test
    void anUnsafeFilenameReturnsAnErrorAndWritesNothing() {
        NotesTools tools = tools(NotesSummarizer.NONE);
        String result = tools.saveToFile("content", "...");

        assertThat(result).contains("not usable");
    }

    // ── sanitize ──────────────────────────────────────────────────────────────

    @Test
    void sanitizeKeepsSafeFilenames() {
        assertThat(NotesTools.sanitize("2026-09-27-notes.md")).isEqualTo("2026-09-27-notes.md");
        assertThat(NotesTools.sanitize("weekly_review.txt")).isEqualTo("weekly_review.txt");
    }

    @Test
    void sanitizeStripsPathSeparators() {
        assertThat(NotesTools.sanitize("../../secret")).isEqualTo("secret");
        assertThat(NotesTools.sanitize("/etc/passwd")).isEqualTo("etcpasswd");
    }

    @Test
    void sanitizeStripsLeadingDots() {
        assertThat(NotesTools.sanitize("...dangerous")).isEqualTo("dangerous");
    }

    @Test
    void sanitizeReturnsEmptyForBlankInput() {
        assertThat(NotesTools.sanitize("")).isEmpty();
        assertThat(NotesTools.sanitize("   ")).isEmpty();
        assertThat(NotesTools.sanitize(null)).isEmpty();
    }

    // ── summarize ─────────────────────────────────────────────────────────────

    @Test
    void summarizeWithNoneReturnsAnErrorMessage() {
        NotesTools tools = tools(NotesSummarizer.NONE);
        String result = tools.summarize("some text", null);

        assertThat(result).contains("failed");
    }

    @Test
    void summarizeWithBlankTextReturnsAnErrorMessage() {
        NotesTools tools = tools(NotesSummarizer.NONE);
        String result = tools.summarize("  ", null);

        assertThat(result).contains("No text");
    }

    @Test
    void summarizePassesResultThrough() {
        NotesSummarizer stub = (text, instructions) -> "# Summary\n- Item one";
        NotesTools tools = tools(stub);

        assertThat(tools.summarize("raw text", "bullet list")).isEqualTo("# Summary\n- Item one");
    }
}
