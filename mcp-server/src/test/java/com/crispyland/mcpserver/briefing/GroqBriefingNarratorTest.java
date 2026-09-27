package com.crispyland.mcpserver.briefing;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.jsonPath;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withTooManyRequests;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.json.JsonMapper;

/**
 * The narrator has one hard rule — <b>it cannot cost a collection</b> — and one setting that is
 * load-bearing rather than tunable.
 * <p>
 * The setting is {@code reasoning_effort}. Left off, gpt-oss spends the whole {@code max_tokens}
 * budget deliberating over a table of seven numbers and answers {@code finish_reason: length} with no
 * content, which reaches the card as a perfectly healthy run that merely has no sentence — a failure
 * with no symptom anywhere. That is why it is asserted on the wire here and not merely defaulted in a
 * record.
 */
class GroqBriefingNarratorTest {

    private static final String ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";

    private static final Briefing DAY = new Briefing(LocalDate.of(2026, 9, 26),
            Instant.parse("2026-09-26T11:02:38Z"), 6, 2, 240, 4, 2,
            List.of("all day — Meeting with the SACE board", "20:30–21:30 Morning run"), "");

    private record Fixture(BriefingNarrator narrator, MockRestServiceServer server) {
    }

    private static BriefingProperties.Narrator settings(String reasoningEffort) {
        return new BriefingProperties.Narrator("gsk-test", ENDPOINT, "gpt-oss-20b", 300, 0.3,
                Duration.ofSeconds(20), reasoningEffort);
    }

    private static Fixture fixture(BriefingProperties.Narrator settings) {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        return new Fixture(
                new GroqBriefingNarrator(builder.build(), JsonMapper.builder().build(), settings),
                server);
    }

    private static Fixture fixture() {
        return fixture(settings("low"));
    }

    private static String reply(String content) {
        return "{\"choices\":[{\"finish_reason\":\"stop\",\"message\":{\"role\":\"assistant\","
                + "\"content\":\"" + content + "\"}}]}";
    }

    @Test
    void theRequestCarriesTheNarratorsOwnModelBudgetAndReasoningEffort() {
        Fixture fixture = fixture();
        fixture.server().expect(requestTo(ENDPOINT))
                .andExpect(jsonPath("$.model").value("gpt-oss-20b"))
                .andExpect(jsonPath("$.max_tokens").value(300))
                .andExpect(jsonPath("$.temperature").value(0.3))
                // The one that has no symptom when it is missing.
                .andExpect(jsonPath("$.reasoning_effort").value("low"))
                // No tools, ever: this is a rewrite of numbers already computed, and a tool schema
                // here would be an unattended job with a way to do something.
                .andExpect(jsonPath("$.tools").doesNotExist())
                .andRespond(withSuccess(reply("Six events today."), MediaType.APPLICATION_JSON));

        assertThat(fixture.narrator().narrate(DAY)).isEqualTo("Six events today.");
        fixture.server().verify();
    }

    /** Some models reject the parameter outright, so blank has to mean absent rather than empty. */
    @Test
    void anEmptyReasoningEffortOmitsTheParameterRatherThanSendingItBlank() {
        Fixture fixture = fixture(settings(""));
        fixture.server().expect(requestTo(ENDPOINT))
                .andExpect(jsonPath("$.reasoning_effort").doesNotExist())
                .andRespond(withSuccess(reply("Six events today."), MediaType.APPLICATION_JSON));

        assertThat(fixture.narrator().narrate(DAY)).isEqualTo("Six events today.");
        fixture.server().verify();
    }

    @Test
    void thePromptCarriesTheFiguresAndTheHighlightsAndNothingElse() {
        Fixture fixture = fixture();
        fixture.server().expect(requestTo(ENDPOINT))
                .andExpect(jsonPath("$.messages[0].role").value("system"))
                .andExpect(jsonPath("$.messages[1].role").value("user"))
                .andExpect(jsonPath("$.messages[1].content").value(org.hamcrest.Matchers
                        .allOf(org.hamcrest.Matchers.containsString("26 September 2026"),
                                org.hamcrest.Matchers.containsString("6 events (2 all-day)"),
                                org.hamcrest.Matchers.containsString("Morning run"))))
                .andExpect(jsonPath("$.messages[2]").doesNotExist())
                .andRespond(withSuccess(reply("Six events today."), MediaType.APPLICATION_JSON));

        fixture.narrator().narrate(DAY);
        fixture.server().verify();
    }

    /**
     * The regression this class exists for. {@code finish_reason: length} with an empty content field
     * is a 200 — the client has to read it as "no sentence" rather than as a sentence.
     */
    @Test
    void aReplyWhoseWholeBudgetWentOnReasoningYieldsNoSentenceRatherThanAnEmptyOne() {
        Fixture fixture = fixture();
        fixture.server().expect(requestTo(ENDPOINT)).andRespond(withSuccess(
                "{\"choices\":[{\"finish_reason\":\"length\",\"message\":{\"role\":\"assistant\","
                        + "\"content\":\"\",\"reasoning\":\"Let me think about this day…\"}}]}",
                MediaType.APPLICATION_JSON));

        assertThat(fixture.narrator().narrate(DAY)).isEmpty();
    }

    @Test
    void noKeyMeansNoCallAtAllRatherThanARejectedOne() {
        Fixture fixture = fixture(new BriefingProperties.Narrator("", ENDPOINT, null, 300, 0.3,
                Duration.ofSeconds(20), "low"));

        assertThat(fixture.narrator().narrate(DAY)).isEmpty();
        // Nothing was expected of the server, and verify() proves nothing was asked of it: a
        // deployment without a key should not be generating 401s once a minute.
        fixture.server().verify();
    }

    @Test
    void aRateLimitYieldsNoSentenceRatherThanAnException() {
        Fixture fixture = fixture();
        fixture.server().expect(requestTo(ENDPOINT)).andRespond(withTooManyRequests());

        assertThat(fixture.narrator().narrate(DAY)).isEmpty();
    }

    @Test
    void aServerErrorYieldsNoSentenceRatherThanAnException() {
        Fixture fixture = fixture();
        fixture.server().expect(requestTo(ENDPOINT)).andRespond(withServerError());

        assertThat(fixture.narrator().narrate(DAY)).isEmpty();
    }

    @Test
    void aBodyThatIsNotTheExpectedShapeYieldsNoSentenceRatherThanAnException() {
        Fixture fixture = fixture();
        fixture.server().expect(requestTo(ENDPOINT))
                .andRespond(withSuccess("<html>502</html>", MediaType.TEXT_HTML));

        assertThat(fixture.narrator().narrate(DAY)).isEmpty();
    }

    @Test
    void anUnreachableProviderYieldsNoSentenceRatherThanAnException() {
        Fixture fixture = fixture();
        fixture.server().expect(requestTo(ENDPOINT)).andRespond(request -> {
            throw new java.net.SocketTimeoutException("Read timed out");
        });

        assertThat(fixture.narrator().narrate(DAY)).isEmpty();
    }

    /** Models like to pad with a newline or a leading space; the card sets this as prose. */
    @Test
    void theSentenceIsStripped() {
        Fixture fixture = fixture();
        fixture.server().expect(requestTo(ENDPOINT))
                .andRespond(withSuccess(reply("\\n  Six events today.  \\n"),
                        MediaType.APPLICATION_JSON));

        assertThat(fixture.narrator().narrate(DAY)).isEqualTo("Six events today.");
    }
}
