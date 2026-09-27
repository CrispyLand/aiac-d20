package com.crispyland.mcpserver.briefing;

import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalTime;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Everything about the job that a deployment gets to decide.
 * <p>
 * {@code schedule} is the <em>initial</em> value only. Once someone picks an option from the
 * dropdown the stored one wins, because a setting a person changed at runtime that reverts on the
 * next deploy is worse than no setting at all. That asymmetry is the reason this is not simply read
 * wherever the trigger is built.
 * <p>
 * {@code file} is held as {@code String} for the same reason as {@code GoogleProperties}: Spring's
 * {@code String → Path} conversion normalizes the value as a resource path and rejects a leading
 * {@code ..}, and {@link Path#of} has no such opinion.
 *
 * @param file     where the snapshots and the run log live. Relative by default, which is correct on
 *                 a laptop and must be overridden to an absolute path under systemd — a unit does
 *                 not start in this directory
 * @param schedule the id of the option to start with when the file has no stored choice
 * @param dailyAt  the local time the {@code daily} option fires at, in the Google zone
 * @param keepRuns how many attempts the log keeps. At the every-minute setting the log grows by 1440
 *                 entries a day and the card shows a handful, so the rest is dropped
 * @param narrator the sentence-writing step's own budget — see {@link Narrator}
 */
@ConfigurationProperties(prefix = "briefing")
public record BriefingProperties(
        String file,
        String schedule,
        LocalTime dailyAt,
        int keepRuns,
        Narrator narrator) {

    public BriefingProperties {
        file = (file == null || file.isBlank()) ? "data/briefings.json" : file.strip();
        schedule = (schedule == null) ? "" : schedule.strip();
        dailyAt = (dailyAt == null) ? LocalTime.of(7, 0) : dailyAt;
        keepRuns = (keepRuns <= 0) ? 20 : keepRuns;
        narrator = (narrator == null) ? new Narrator(null, null, null, 0, -1, null, null) : narrator;
    }

    public Path path() {
        return Path.of(file).toAbsolutePath().normalize();
    }

    /** Unrecognised resolves to {@code OFF}: a bad value in the YAML should not run the job on a guess. */
    public BriefingSchedule initialSchedule() {
        return BriefingSchedule.from(schedule);
    }

    /**
     * The narrator's own model, key and ceiling — deliberately not the agent's.
     * <p>
     * Turning seven numbers into one sentence is a mechanical rewrite, not a reasoning task, so it
     * gets the small model and a tight token cap. Separate settings also mean the unattended job
     * cannot quietly inherit an expensive model someone chose for the chat.
     *
     * @param apiKey      supplied by the environment, never committed. Blank is a supported state:
     *                    the figures are still collected and the run is logged as not narrated
     * @param endpoint    the OpenAI-compatible chat-completions URL
     * @param model       small on purpose
     * @param maxTokens   a sentence or two, and a hard stop on what a pathological day can spend.
     *                    On a reasoning model the hidden reasoning tokens are billed against this
     *                    same ceiling and spent <em>first</em>, so it has to cover both — see
     *                    {@code reasoningEffort}
     * @param temperature low, because the prose should describe the figures rather than embroider
     *                    them
     * @param timeout     the whole call. Short: a scheduled job waiting on a hung provider is a job
     *                    that is not collecting
     * @param reasoningEffort {@code low} by default, and not a tuning knob — without it gpt-oss
     *                    spends the entire {@code maxTokens} thinking about a table of seven numbers
     *                    and returns {@code finish_reason: length} with no content at all, which
     *                    reaches the card as a perfectly healthy run that simply has no sentence.
     *                    {@code ""} omits the parameter, which some models (qwen/compound) require
     */
    public record Narrator(
            String apiKey,
            String endpoint,
            String model,
            int maxTokens,
            double temperature,
            Duration timeout,
            String reasoningEffort) {

        public Narrator {
            apiKey = (apiKey == null) ? "" : apiKey.strip();
            endpoint = (endpoint == null || endpoint.isBlank())
                    ? "https://api.cerebras.ai/v1/chat/completions" : endpoint.strip();
            model = (model == null || model.isBlank()) ? "gpt-oss-120b" : model.strip();
            maxTokens = (maxTokens <= 0) ? 300 : maxTokens;
            temperature = (temperature < 0) ? 0.3 : temperature;
            timeout = (timeout == null) ? Duration.ofSeconds(20) : timeout;
            reasoningEffort = (reasoningEffort == null) ? "low" : reasoningEffort.strip();
        }

        public boolean hasKey() {
            return !apiKey.isEmpty();
        }
    }
}
