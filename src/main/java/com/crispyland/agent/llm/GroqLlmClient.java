package com.crispyland.agent.llm;

import com.crispyland.agent.memory.Message;
import com.crispyland.agent.usage.TokenUsage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import tools.jackson.core.JacksonException;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * The single class in the whole application that performs HTTP and touches JSON.
 * Speaks the OpenAI-compatible chat-completions dialect that Groq exposes.
 */
public class GroqLlmClient implements LlmClient {

    private final RestClient restClient;
    private final ObjectMapper mapper;
    private final String endpoint;
    private final String apiKey;

    public GroqLlmClient(RestClient restClient, ObjectMapper mapper, String endpoint, String apiKey) {
        this.restClient = restClient;
        this.mapper = mapper;
        this.endpoint = endpoint;
        this.apiKey = apiKey;
    }

    @Override
    public ChatResponse complete(ChatRequest request) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new LlmException("CEREBRAS_API_KEY is not set — export it and restart the app.");
        }

        String payload = mapper.writeValueAsString(toJson(request));
        ResponseEntity<String> response;
        try {
            response = restClient.post()
                    .uri(endpoint)
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
                    .contentType(MediaType.APPLICATION_JSON)
                    .accept(MediaType.APPLICATION_JSON)
                    .body(payload)
                    .retrieve()
                    // Disable the default throw-on-error so we can surface status + body ourselves.
                    .onStatus(status -> true, (req, res) -> { })
                    .toEntity(String.class);
        } catch (RestClientException e) {
            throw new LlmException("Could not reach LLM API at " + endpoint + ": " + e.getMessage(), e);
        }

        String body = response.getBody();
        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new LlmException("LLM API returned HTTP %s: %s"
                    .formatted(response.getStatusCode().value(), describeError(body)));
        }
        return parse(body);
    }

    /**
     * Error bodies are {@code {"error": {"message": ..., "code": ...}}}. Lifting the message
     * out matters most for {@code context_length_exceeded}: that is the provider telling you
     * the prompt did not fit, and it is worth reading rather than scrolling past as JSON.
     */
    private String describeError(String body) {
        if (body == null || body.isBlank()) {
            return "<empty body>";
        }
        try {
            JsonNode error = mapper.readTree(body).path("error");
            String message = text(error.path("message"));
            if (!message.isBlank()) {
                String code = text(error.path("code"));
                return code.isBlank() ? message : message + " [" + code + "]";
            }
        } catch (JacksonException e) {
            // Not JSON — fall through and show whatever came back.
        }
        return summarize(body);
    }

    private ObjectNode toJson(ChatRequest request) {
        ObjectNode root = mapper.createObjectNode();
        root.put("model", request.model());

        ArrayNode messages = root.putArray("messages");
        for (Message message : request.messages()) {
            ObjectNode node = messages.addObject();
            node.put("role", message.role());
            node.put("content", message.content());
        }
        // Completed tool exchanges go on the end, in order, and the shape is dictated: the
        // assistant message that requested the calls must be replayed before the results, or the
        // provider rejects a tool_call_id it has no matching request for.
        for (ToolRound round : request.rounds()) {
            ObjectNode asked = messages.addObject();
            asked.put("role", Message.ASSISTANT);
            asked.putNull("content");
            ArrayNode calls = asked.putArray("tool_calls");
            for (ToolCall call : round.calls()) {
                ObjectNode entry = calls.addObject();
                entry.put("id", call.id());
                entry.put("type", "function");
                ObjectNode function = entry.putObject("function");
                function.put("name", call.name());
                // Back to a string, because that is how the provider sent it and how it expects
                // to see it echoed. The map in between is for our benefit, not the wire's.
                function.put("arguments", mapper.writeValueAsString(call.arguments()));
            }
            for (ToolResult result : round.results()) {
                ObjectNode answered = messages.addObject();
                answered.put("role", "tool");
                answered.put("tool_call_id", result.callId());
                answered.put("content", result.content());
            }
        }

        // Absent entirely when nothing is offered. See ChatRequest#tools: this is what keeps a
        // switched-off turn identical to the turns this app made before tools existed.
        if (!request.tools().isEmpty()) {
            ArrayNode tools = root.putArray("tools");
            for (ToolSpec spec : request.tools()) {
                ObjectNode entry = tools.addObject();
                entry.put("type", "function");
                ObjectNode function = entry.putObject("function");
                function.put("name", spec.name());
                function.put("description", spec.description());
                function.set("parameters", mapper.valueToTree(spec.parameters()));
            }
        }

        if (request.temperature() != null) {
            root.put("temperature", request.temperature());
        }
        if (request.maxCompletionTokens() != null) {
            root.put("max_completion_tokens", request.maxCompletionTokens());
        }
        if (request.reasoningEffort() != null && !request.reasoningEffort().isBlank()) {
            root.put("reasoning_effort", request.reasoningEffort());
        }
        if (!request.stopSequences().isEmpty()) {
            ArrayNode stop = root.putArray("stop");
            request.stopSequences().forEach(stop::add);
        }
        if (request.responseSchema() != null && !request.responseSchema().isBlank()) {
            ObjectNode format = root.putObject("response_format");
            format.put("type", "json_schema");
            ObjectNode schema = format.putObject("json_schema");
            schema.put("name", "response");
            schema.put("strict", true);
            try {
                schema.set("schema", mapper.readTree(request.responseSchema()));
            } catch (JacksonException e) {
                throw new LlmException("Response JSON schema is not valid JSON: " + e.getOriginalMessage(), e);
            }
        }
        return root;
    }

    private ChatResponse parse(String body) {
        JsonNode root;
        try {
            root = mapper.readTree(body);
        } catch (JacksonException e) {
            throw new LlmException("Groq returned a body that is not JSON: " + summarize(body), e);
        }

        JsonNode choice = root.path("choices").path(0);
        if (choice.isMissingNode()) {
            throw new LlmException("Groq response contained no choices: " + summarize(body));
        }
        JsonNode message = choice.path("message");
        String content = text(message.path("content"));
        if (content.isBlank()) {
            // gpt-oss models can spend the whole budget on reasoning and return empty content.
            content = text(message.path("reasoning"));
        }

        JsonNode usage = root.path("usage");
        TokenUsage tokenUsage = new TokenUsage(
                number(usage.path("prompt_tokens")),
                number(usage.path("completion_tokens")),
                number(usage.path("total_tokens")));

        return new ChatResponse(content, text(root.path("model")),
                text(choice.path("finish_reason")), tokenUsage, toolCalls(message));
    }

    /**
     * The calls the model wants made, if any.
     * <p>
     * {@code arguments} is a JSON string the <em>model</em> wrote, so it is the least trustworthy
     * JSON in the application: it is generated text that is only usually valid. Parsing it here
     * means a malformed one is caught at the boundary and reported as a bad call, instead of
     * reaching the code that runs tools and failing there as something less explicable.
     */
    private List<ToolCall> toolCalls(JsonNode message) {
        JsonNode calls = message.path("tool_calls");
        if (!calls.isArray() || calls.isEmpty()) {
            return List.of();
        }
        List<ToolCall> parsed = new ArrayList<>();
        for (JsonNode call : calls) {
            JsonNode function = call.path("function");
            String name = text(function.path("name"));
            String arguments = text(function.path("arguments"));
            Map<String, Object> values;
            try {
                values = arguments.isBlank() ? Map.of()
                        : mapper.readValue(arguments, new TypeReference<Map<String, Object>>() { });
            } catch (JacksonException e) {
                throw new LlmException("The model asked to call '%s' with arguments that are not "
                        + "valid JSON: %s".formatted(name, summarize(arguments)), e);
            }
            parsed.add(new ToolCall(text(call.path("id")), name, values));
        }
        return parsed;
    }

    private static String text(JsonNode node) {
        return node.isTextual() ? node.stringValue() : "";
    }

    private static long number(JsonNode node) {
        return node.isNumber() ? node.longValue() : 0L;
    }

    private static String summarize(String body) {
        if (body == null || body.isBlank()) {
            return "<empty body>";
        }
        String trimmed = body.strip();
        return trimmed.length() > 600 ? trimmed.substring(0, 600) + "…" : trimmed;
    }
}
