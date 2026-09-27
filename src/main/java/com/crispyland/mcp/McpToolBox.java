package com.crispyland.mcp;

import com.crispyland.agent.llm.ToolBox;
import com.crispyland.agent.llm.ToolCall;
import com.crispyland.agent.llm.ToolResult;
import com.crispyland.agent.llm.ToolSpec;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.spec.McpSchema;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * The discovered tools, offered to the model and run on its behalf.
 * <p>
 * This is the class that changes what the agent <em>is</em>. Everything else about MCP so far has
 * been observation — a panel reporting what another process claims it can do. Here the model is
 * told about capabilities that did not exist when it was trained and do not exist in this
 * codebase, and is allowed to use them. Nothing in this file mentions currencies, weather or
 * news: it would work unchanged against a server nobody has written yet, which is the entire
 * argument for the protocol.
 * <p>
 * The switch on the page governs this too, and deliberately so. Off means the model is not told
 * the tools exist, so it cannot call them, so a chat turn goes out exactly as it did before any
 * of this — the difference is visible in one click rather than in a rebuild.
 */
@Service
public class McpToolBox implements ToolBox {

    private static final Logger log = LoggerFactory.getLogger(McpToolBox.class);

    private final McpCatalogue mcp;

    /**
     * Which connection owns which tool name, rebuilt by every {@link #available()}.
     * <p>
     * Needed because the model calls a bare name and the protocol needs a specific server. Being
     * rebuilt each turn rather than cached is what lets a server appear or vanish between messages
     * without leaving a name pointing at a client that can no longer serve it.
     */
    private volatile Map<String, McpSyncClient> owners = Map.of();

    public McpToolBox(McpCatalogue mcp) {
        this.mcp = mcp;
    }

    @Override
    public List<ToolSpec> available() {
        Map<String, McpSyncClient> byName = new LinkedHashMap<>();
        List<ToolSpec> specs = new ArrayList<>();
        for (McpSyncClient client : mcp.connections()) {
            String server = McpCatalogue.label(client);
            try {
                if (!client.isInitialized()) {
                    client.initialize();
                }
                for (McpSchema.Tool tool : client.listTools().tools()) {
                    String name = tool.name();
                    // First server to claim a name keeps it. The alternative — silently letting
                    // the last one win — means an unrelated server coming online can change what
                    // an existing tool does, which is a supply-chain problem, not a naming one.
                    if (byName.putIfAbsent(name, client) != null) {
                        log.warn("Two MCP servers both offer '{}'; keeping the one from '{}' and "
                                + "ignoring '{}'.", name, McpCatalogue.label(byName.get(name)), server);
                        continue;
                    }
                    specs.add(new ToolSpec(name, describe(tool), tool.inputSchema()));
                }
            } catch (Exception e) {
                // A server that will not answer contributes no tools and stops nothing. The panel
                // is already saying it is down; the turn should still be answerable without it.
                log.warn("MCP server '{}' offered no tools this turn: {}", server, e.toString());
            }
        }
        owners = byName;
        return List.copyOf(specs);
    }

    @Override
    public ToolResult call(ToolCall call) {
        McpSyncClient client = owners.get(call.name());
        if (client == null) {
            // Reachable: the model can invent a name, or call one from a server that has since
            // gone away. Told plainly, it usually apologises and answers without the tool.
            log.warn("The model asked for tool '{}', which no connected server offers.", call.name());
            return ToolResult.failed(call, "No connected MCP server offers a tool called '%s'."
                    .formatted(call.name()), "");
        }
        String server = McpCatalogue.label(client);
        try {
            McpSchema.CallToolResult result = client.callTool(
                    McpSchema.CallToolRequest.builder()
                            .name(call.name())
                            .arguments(call.arguments())
                            .build());

            String text = textOf(result);
            // Logged at INFO because this is the line that proves the claim: a tool ran, in
            // another process, because the model decided it should.
            log.info("MCP tool '{}' on '{}' called with {} -> {}", call.name(), server,
                    call.arguments(), clip(text));

            // isError is the server saying "your call was wrong", which is a different thing from
            // the call failing to happen. Both go back to the model; only this one is its fault.
            return Boolean.TRUE.equals(result.isError())
                    ? ToolResult.failed(call, text, server)
                    : ToolResult.of(call, text, server);
        } catch (Exception e) {
            log.warn("MCP tool '{}' on '{}' failed: {}", call.name(), server, e.toString());
            return ToolResult.failed(call, "The tool could not be run: " + e, server);
        }
    }

    /**
     * Everything the server says about the tool, because the description is the only thing
     * steering the model's choice. A title is often the readable half of it and costs a few
     * tokens to include.
     */
    private static String describe(McpSchema.Tool tool) {
        String title = (tool.title() == null) ? "" : tool.title().strip();
        String description = (tool.description() == null) ? "" : tool.description().strip();
        if (title.isEmpty() || description.startsWith(title)) {
            return description;
        }
        return description.isEmpty() ? title : title + ". " + description;
    }

    /**
     * The result as the model will read it.
     * <p>
     * Only text blocks. The protocol also allows images and embedded resources, and silently
     * dropping those is the honest behaviour for a chat completion that can only receive a string
     * — better an empty result the model can react to than a caption pretending to be the content.
     */
    private static String textOf(McpSchema.CallToolResult result) {
        StringBuilder text = new StringBuilder();
        for (McpSchema.Content content : result.content()) {
            if (content instanceof McpSchema.TextContent block && block.text() != null) {
                if (!text.isEmpty()) {
                    text.append('\n');
                }
                text.append(block.text());
            }
        }
        return text.isEmpty() ? "(the tool returned nothing)" : text.toString();
    }

    private static String clip(String text) {
        String one = text.replace('\n', ' ').strip();
        return one.length() > 160 ? one.substring(0, 160) + "…" : one;
    }
}
