package com.crispyland.mcpserver.notes;

import java.nio.file.Path;
import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Where notes are saved and what the summarizer is allowed to spend.
 *
 * @param directory where {@code saveToFile} writes. Relative by default — must be overridden to
 *                  an absolute path under systemd, same reason as {@code briefing.file}
 * @param groq      the model call that {@code summarize} makes
 */
@ConfigurationProperties(prefix = "notes")
public record NotesProperties(String directory, Groq groq) {

    public NotesProperties {
        directory = (directory == null || directory.isBlank()) ? "./notes" : directory.strip();
        groq = (groq == null) ? new Groq(null, null, null, 0, -1, null, null) : groq;
    }

    public Path path() {
        return Path.of(directory).toAbsolutePath().normalize();
    }

    /**
     * @param apiKey          shared with the briefing narrator via {@code ${GROQ_API_KEY:}}
     * @param maxTokens       higher than the briefing narrator — notes can be several paragraphs
     * @param reasoningEffort {@code low} for the same reason as the narrator: without it
     *                        {@code gpt-oss} spends the token budget deliberating and returns
     *                        nothing
     */
    public record Groq(
            String apiKey,
            String endpoint,
            String model,
            int maxTokens,
            double temperature,
            Duration timeout,
            String reasoningEffort) {

        public Groq {
            apiKey = (apiKey == null) ? "" : apiKey.strip();
            endpoint = (endpoint == null || endpoint.isBlank())
                    ? "https://api.cerebras.ai/v1/chat/completions" : endpoint.strip();
            model = (model == null || model.isBlank()) ? "gpt-oss-20b" : model.strip();
            maxTokens = (maxTokens <= 0) ? 1000 : maxTokens;
            temperature = (temperature < 0) ? 0.5 : temperature;
            timeout = (timeout == null) ? Duration.ofSeconds(30) : timeout;
            reasoningEffort = (reasoningEffort == null) ? "low" : reasoningEffort.strip();
        }

        public boolean hasKey() {
            return !apiKey.isEmpty();
        }
    }
}
