package com.crispyland.mcpserver.notes;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.mcp.annotation.McpTool;
import org.springframework.ai.mcp.annotation.McpToolParam;
import org.springframework.stereotype.Service;

/**
 * The two tools that form the processing and persistence steps of the pipeline.
 * <p>
 * Neither tool knows about the other. The model decides the order: it calls {@link #summarize}
 * with whatever text it has assembled, then passes the result to {@link #saveToFile}. The
 * composition is at the agent level — these are building blocks, not a fixed sequence.
 * <p>
 * {@link #summarize} returns the result rather than saving it so the model can inspect,
 * modify, or discard it before committing to disk. A tool that fetched, summarised and saved
 * in one call would make that judgment for the model, which is the one judgment worth leaving
 * to it.
 */
@Service
public class NotesTools {

    private static final Logger log = LoggerFactory.getLogger(NotesTools.class);

    /** A filename that would escape the notes directory or overwrite something unexpected. */
    private static final int MAX_FILENAME_LENGTH = 120;

    private final NotesSummarizer summarizer;
    private final Path notesDir;

    public NotesTools(NotesSummarizer summarizer, NotesProperties properties) {
        this.summarizer = summarizer;
        this.notesDir = properties.path();
    }

    @McpTool(name = "summarize",
            description = "Process or summarize text using AI. Pass the raw output of other "
                    + "tools — getSchedule, getTasks, or any text — and optionally describe "
                    + "what to produce (e.g. 'structured meeting agenda', 'bullet-point action "
                    + "list', 'weekly review'). Returns the processed text. Read-only — does "
                    + "not save anything. If the user asked to save the result, you MUST call "
                    + "saveToFile immediately after this with the returned text.")
    public String summarize(
            @McpToolParam(description = "The text to process.", required = true)
            String text,
            @McpToolParam(description = "How to process it — e.g. 'structured daily agenda "
                    + "in markdown', 'action items only', 'weekly summary'. Omit for a plain "
                    + "summary.", required = false)
            String instructions) {

        if (text == null || text.isBlank()) {
            return "No text was provided.";
        }
        log.info("summarize: {} chars, instructions: '{}'", text.length(),
                instructions == null ? "(none)" : instructions);

        String result = summarizer.summarize(text, instructions);
        if (result.isEmpty()) {
            return "Summarization failed — no API key configured or the call did not succeed.";
        }
        return result;
    }

    @McpTool(name = "saveToFile",
            description = "Save text content to a file in the server's notes directory. "
                    + "Returns the absolute path of the saved file. Use after summarize to "
                    + "persist a result, or to save any text the conversation produced.")
    public String saveToFile(
            @McpToolParam(description = "The content to write.", required = true)
            String content,
            @McpToolParam(description = "The filename, including extension — e.g. "
                    + "'2026-09-27-notes.md'. Path separators are stripped; only the "
                    + "filename part is used.", required = true)
            String filename) {

        if (content == null || content.isBlank()) {
            return "No content was provided — nothing was saved.";
        }

        String safe = sanitize(filename);
        if (safe.isEmpty()) {
            return "The filename '" + filename + "' is not usable after removing unsafe "
                    + "characters. Provide a plain filename such as '2026-09-27-notes.md'.";
        }

        try {
            Files.createDirectories(notesDir);
            Path target = notesDir.resolve(safe);
            Files.writeString(target, content);
            log.info("saveToFile: wrote {} chars to {}", content.length(), target);
            return "Saved to " + target;
        } catch (IOException e) {
            throw new UncheckedIOException("Could not write to " + notesDir.resolve(safe), e);
        }
    }

    /**
     * Strip anything that would let the filename escape the notes directory or silently
     * overwrite something. Keeps letters, digits, dots, hyphens and underscores.
     */
    static String sanitize(String filename) {
        if (filename == null || filename.isBlank()) {
            return "";
        }
        // Keep only safe characters — this also removes / \ and ..
        String safe = filename.strip().replaceAll("[^A-Za-z0-9._\\-]", "");
        // A name that is all dots would still be a problem (.., ...)
        safe = safe.replaceAll("^\\.+", "");
        if (safe.length() > MAX_FILENAME_LENGTH) {
            safe = safe.substring(0, MAX_FILENAME_LENGTH);
        }
        return safe;
    }
}
