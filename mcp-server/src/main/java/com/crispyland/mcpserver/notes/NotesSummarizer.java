package com.crispyland.mcpserver.notes;

/**
 * Processes or summarizes text via an AI model.
 * <p>
 * Same contract as {@link com.crispyland.mcpserver.briefing.BriefingNarrator}: implementations
 * return {@code ""} rather than throwing on failure. The tool layer converts an empty result into
 * an error message so the model knows the step failed.
 */
@FunctionalInterface
public interface NotesSummarizer {

    /**
     * Process {@code text} according to {@code instructions}, or summarize it if instructions are
     * absent. Returns the result, or {@code ""} on any failure including no API key.
     */
    String summarize(String text, String instructions);

    /** For a deployment with no key configured: honest about producing nothing, and free. */
    NotesSummarizer NONE = (text, instructions) -> "";
}
