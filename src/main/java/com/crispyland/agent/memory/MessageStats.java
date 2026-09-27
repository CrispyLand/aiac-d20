package com.crispyland.agent.memory;

import java.util.List;

/**
 * What one message cost, captured when it was recorded.
 * <p>
 * The provider reports usage per turn, not per message, so the turn's numbers are split
 * the way they were actually earned: the user message carries the prompt tokens (the whole
 * context replayed to produce it), the assistant message carries the completion tokens plus
 * the turn total, latency and finish reason.
 *
 * @param toolsUsed  which tools ran, in order. Kept for backwards-compatible JSON deserialization
 *                   of stored messages; derived from {@code toolSteps} on new turns.
 * @param toolSteps  richer version of {@code toolsUsed}: each step also carries the tool's result
 *                   text, used by the chat view to show the chain and offer download links.
 *                   Empty on messages written before this field existed.
 */
public record MessageStats(
        long promptTokens,
        long completionTokens,
        long totalTokens,
        long latencyMillis,
        String model,
        String finishReason,
        List<String> toolsUsed,
        List<ToolStep> toolSteps) {

    /**
     * One step in a multi-tool chain: what was asked and what came back.
     *
     * @param server the MCP server label, e.g. "calendar" or "weather". Empty string when unknown.
     */
    public record ToolStep(String name, String result, boolean failed, String server) {

        /**
         * The bare filename of a file saved by {@code saveToFile}, or {@code ""} if this step is
         * not a successful save. Used by the template to build a download link without fragile
         * string-splitting in Thymeleaf.
         */
        public String savedFilename() {
            if (!"saveToFile".equals(name) || failed) {
                return "";
            }
            if (!result.startsWith("Saved to ")) {
                return "";
            }
            String path = result.substring("Saved to ".length()).strip();
            int slash = path.lastIndexOf('/');
            return slash >= 0 ? path.substring(slash + 1) : path;
        }
    }

    public MessageStats {
        toolsUsed = (toolsUsed == null) ? List.of() : List.copyOf(toolsUsed);
        toolSteps = (toolSteps == null) ? List.of() : List.copyOf(toolSteps);
    }

    public static MessageStats forPrompt(long promptTokens, String model) {
        return new MessageStats(promptTokens, 0, 0, 0, model, "", List.of(), List.of());
    }

    public static MessageStats forCompletion(long completionTokens, long totalTokens,
                                             long latencyMillis, String model, String finishReason) {
        return new MessageStats(0, completionTokens, totalTokens, latencyMillis, model,
                finishReason, List.of(), List.of());
    }

    /**
     * Records the rich tool steps. Also derives and sets {@code toolsUsed} so the JSON written for
     * this turn carries both — the view can use either, and the field is always consistent.
     */
    public MessageStats withToolSteps(List<ToolStep> steps) {
        List<String> names = steps.stream()
                .map(s -> s.failed() ? s.name() + " (failed)" : s.name())
                .toList();
        return new MessageStats(promptTokens, completionTokens, totalTokens, latencyMillis,
                model, finishReason, names, steps);
    }

    /** @deprecated use {@link #withToolSteps} for new code; kept for any call sites not yet migrated */
    public MessageStats withToolsUsed(List<String> tools) {
        return new MessageStats(promptTokens, completionTokens, totalTokens, latencyMillis,
                model, finishReason, tools, List.of());
    }
}
