package com.crispyland.mcpserver.notes;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * A general-purpose Groq call: text in, processed text out.
 * <p>
 * Same failure contract as {@code GroqBriefingNarrator}: returns {@code ""} rather than throwing.
 * The tool layer turns an empty result into an explicit error message so the model knows the step
 * failed rather than succeeded with no output.
 */
public class GroqNotesSummarizer implements NotesSummarizer {

    private static final Logger log = LoggerFactory.getLogger(GroqNotesSummarizer.class);

    private static final String SYSTEM = """
            You are a helpful assistant. Process the provided text according to the instructions \
            given. Return plain text or markdown as appropriate.
            Rules:
            - Follow the instructions exactly.
            - No greeting, no sign-off, no meta-commentary about what you are doing.
            - If no instructions are given, produce a clear, concise summary.""";

    private final RestClient restClient;
    private final ObjectMapper mapper;
    private final NotesProperties.Groq settings;

    public GroqNotesSummarizer(RestClient restClient, ObjectMapper mapper,
                               NotesProperties.Groq settings) {
        this.restClient = restClient;
        this.mapper = mapper;
        this.settings = settings;
    }

    /**
     * Process {@code text} according to {@code instructions}. Returns the model's response, or
     * {@code ""} on any failure.
     */
    @Override
    public String summarize(String text, String instructions) {
        if (!settings.hasKey()) {
            log.debug("No notes API key configured — returning empty.");
            return "";
        }
        try {
            ResponseEntity<String> response = restClient.post()
                    .uri(settings.endpoint())
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + settings.apiKey())
                    .contentType(MediaType.APPLICATION_JSON)
                    .accept(MediaType.APPLICATION_JSON)
                    .body(mapper.writeValueAsString(request(text, instructions)))
                    .retrieve()
                    .onStatus(status -> true, (req, res) -> { })
                    .toEntity(String.class);

            if (!response.getStatusCode().is2xxSuccessful()) {
                log.warn("Notes summarizer got HTTP {} — returning empty.",
                        response.getStatusCode().value());
                return "";
            }
            return content(response.getBody());
        } catch (Exception e) {
            log.warn("Notes summarizer failed ({}: {}) — returning empty.",
                    e.getClass().getSimpleName(), e.getMessage());
            return "";
        }
    }

    private ObjectNode request(String text, String instructions) {
        ObjectNode root = mapper.createObjectNode();
        root.put("model", settings.model());
        root.put("temperature", settings.temperature());
        root.put("max_tokens", settings.maxTokens());
        if (!settings.reasoningEffort().isEmpty()) {
            root.put("reasoning_effort", settings.reasoningEffort());
        }

        ArrayNode messages = root.putArray("messages");
        addMessage(messages, "system", SYSTEM);

        // Instructions first so the model reads "what to do" before "what to do it to".
        String userContent = (instructions == null || instructions.isBlank())
                ? text
                : instructions.strip() + "\n\n" + text;
        addMessage(messages, "user", userContent);
        return root;
    }

    private void addMessage(ArrayNode messages, String role, String content) {
        ObjectNode node = messages.addObject();
        node.put("role", role);
        node.put("content", content);
    }

    private String content(String body) {
        if (body == null || body.isBlank()) {
            return "";
        }
        JsonNode message = mapper.readTree(body).path("choices").path(0).path("message");
        return message.path("content").stringValue("").strip();
    }
}
