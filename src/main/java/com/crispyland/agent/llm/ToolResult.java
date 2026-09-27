package com.crispyland.agent.llm;

/**
 * What running a tool produced, on its way back to the model.
 *
 * @param failed whether the tool refused or broke. The text still goes back either way, and that
 *               is the point: a model told "that currency code does not exist" can correct itself
 *               and ask again, whereas a turn aborted on the exception just loses. This flag
 *               exists so the failure can be logged and shown as a failure, not so it can be
 *               hidden from the model
 * @param server the MCP server label that ran this tool, e.g. "calendar" or "weather". Empty
 *               string when the origin is unknown (e.g. the no-op {@link ToolBox}).
 */
public record ToolResult(String callId, String name, String content, boolean failed, String server) {

    public static ToolResult of(ToolCall call, String content, String server) {
        return new ToolResult(call.id(), call.name(), content, false, server);
    }

    public static ToolResult failed(ToolCall call, String problem, String server) {
        return new ToolResult(call.id(), call.name(), problem, true, server);
    }

    /** Backwards-compatible overload for call sites that do not know the server. */
    public static ToolResult of(ToolCall call, String content) {
        return of(call, content, "");
    }

    /** Backwards-compatible overload for call sites that do not know the server. */
    public static ToolResult failed(ToolCall call, String problem) {
        return failed(call, problem, "");
    }
}
