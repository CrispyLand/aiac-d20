package com.crispyland.mcpserver.notes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.jsonPath;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import java.time.Duration;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.json.JsonMapper;

class NotesSummarizerTest {

    private static final String ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";

    private record Fixture(GroqNotesSummarizer summarizer, MockRestServiceServer server) {
    }

    private static NotesProperties.Groq settings(String apiKey) {
        return new NotesProperties.Groq(apiKey, ENDPOINT, "openai/gpt-oss-20b", 1000, 0.5,
                Duration.ofSeconds(30), "low");
    }

    private static Fixture fixture(String apiKey) {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        return new Fixture(
                new GroqNotesSummarizer(builder.build(), JsonMapper.builder().build(),
                        settings(apiKey)),
                server);
    }

    private static String reply(String content) {
        return "{\"choices\":[{\"finish_reason\":\"stop\",\"message\":{\"role\":\"assistant\","
                + "\"content\":\"" + content + "\"}}]}";
    }

    // ── no key ────────────────────────────────────────────────────────────────

    @Test
    void noKeyMeansNoCallAndEmptyResult() {
        Fixture fixture = fixture("");
        assertThat(fixture.summarizer().summarize("some text", null)).isEmpty();
        fixture.server().verify();  // nothing was called
    }

    @Test
    void noneReturnsEmpty() {
        assertThat(NotesSummarizer.NONE.summarize("some text", null)).isEmpty();
    }

    // ── request shape ─────────────────────────────────────────────────────────

    @Test
    void theRequestCarriesModelBudgetAndReasoningEffort() {
        Fixture fixture = fixture("gsk-test");
        fixture.server().expect(requestTo(ENDPOINT))
                .andExpect(jsonPath("$.model").value("openai/gpt-oss-20b"))
                .andExpect(jsonPath("$.max_tokens").value(1000))
                .andExpect(jsonPath("$.temperature").value(0.5))
                .andExpect(jsonPath("$.reasoning_effort").value("low"))
                .andExpect(jsonPath("$.tools").doesNotExist())
                .andRespond(withSuccess(reply("Summary here."), MediaType.APPLICATION_JSON));

        assertThat(fixture.summarizer().summarize("raw text", null)).isEqualTo("Summary here.");
        fixture.server().verify();
    }

    @Test
    void whenInstructionsAreGivenTheyPrecedeTheTextInTheUserMessage() {
        Fixture fixture = fixture("gsk-test");
        fixture.server().expect(requestTo(ENDPOINT))
                .andExpect(jsonPath("$.messages[1].content")
                        .value(org.hamcrest.Matchers.allOf(
                                org.hamcrest.Matchers.startsWith("bullet list"),
                                org.hamcrest.Matchers.containsString("raw text"))))
                .andRespond(withSuccess(reply("- Item"), MediaType.APPLICATION_JSON));

        fixture.summarizer().summarize("raw text", "bullet list");
        fixture.server().verify();
    }

    @Test
    void withNoInstructionsTheUserMessageIsJustTheText() {
        Fixture fixture = fixture("gsk-test");
        fixture.server().expect(requestTo(ENDPOINT))
                .andExpect(jsonPath("$.messages[1].content").value("raw text"))
                .andRespond(withSuccess(reply("Summary."), MediaType.APPLICATION_JSON));

        fixture.summarizer().summarize("raw text", null);
        fixture.server().verify();
    }

    @Test
    void blankInstructionsTreatedSameAsNull() {
        Fixture fixture = fixture("gsk-test");
        fixture.server().expect(requestTo(ENDPOINT))
                .andExpect(jsonPath("$.messages[1].content").value("raw text"))
                .andRespond(withSuccess(reply("Summary."), MediaType.APPLICATION_JSON));

        fixture.summarizer().summarize("raw text", "   ");
        fixture.server().verify();
    }

    // ── failures ──────────────────────────────────────────────────────────────

    @Test
    void aServerErrorReturnsEmptyRatherThanThrowing() {
        Fixture fixture = fixture("gsk-test");
        fixture.server().expect(requestTo(ENDPOINT)).andRespond(withServerError());

        assertThat(fixture.summarizer().summarize("text", null)).isEmpty();
    }

    @Test
    void aNetworkErrorReturnsEmptyRatherThanThrowing() {
        Fixture fixture = fixture("gsk-test");
        fixture.server().expect(requestTo(ENDPOINT)).andRespond(request -> {
            throw new java.net.SocketTimeoutException("Read timed out");
        });

        assertThat(fixture.summarizer().summarize("text", null)).isEmpty();
    }

    @Test
    void aMalformedBodyReturnsEmpty() {
        Fixture fixture = fixture("gsk-test");
        fixture.server().expect(requestTo(ENDPOINT))
                .andRespond(withSuccess("<html>502</html>", MediaType.TEXT_HTML));

        assertThat(fixture.summarizer().summarize("text", null)).isEmpty();
    }

    // ── response parsing ──────────────────────────────────────────────────────

    @Test
    void theResponseIsStripped() {
        Fixture fixture = fixture("gsk-test");
        fixture.server().expect(requestTo(ENDPOINT))
                .andRespond(withSuccess(reply("\\n  Summary.  \\n"), MediaType.APPLICATION_JSON));

        assertThat(fixture.summarizer().summarize("text", null)).isEqualTo("Summary.");
    }
}
