package com.crispyland.mcpserver.briefing;

import java.time.LocalDate;
import java.time.format.DateTimeParseException;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.mcp.annotation.McpTool;
import org.springframework.ai.mcp.annotation.McpToolParam;
import org.springframework.stereotype.Service;

/**
 * The two tools that read what the job wrote — and the only two answers in this server that Google
 * cannot give directly.
 * <p>
 * {@code getSchedule} already reports what is on a day. {@link #getBriefing} is different in two
 * ways: it answers from the file rather than the network, so it reports the day <em>as it was
 * measured</em>, sentence included; and {@link #getTrend} answers a question with no API behind it
 * at all. Google will say what is on next Tuesday; it will not say that this week was forty minutes
 * busier than last, because nobody recorded last week. That aggregation is the only thing here that
 * is a consequence of having stored anything, rather than a convenience on top of it.
 * <p>
 * Prose out, like the other tools and for the same reason: the consumer is a model reading a string,
 * and a JSON snapshot would be re-read as text at a worse token price. Deliberately, neither tool
 * offers any way to change the schedule or force a run. Those are human controls and they live on the
 * admin API — handing the model a config switch and a button that spends money is a different
 * feature, and not one that was asked for.
 */
@Service
public class BriefingTools {

    private static final Logger log = LoggerFactory.getLogger(BriefingTools.class);

    /** A fortnight of context at most. Beyond that the prose stops being a summary. */
    private static final int MAX_DAYS = 14;

    private final BriefingStore store;
    private final BriefingCollector collector;

    public BriefingTools(BriefingStore store, BriefingCollector collector) {
        this.store = store;
        this.collector = collector;
    }

    // @McpTool disabled — not part of the notes pipeline; re-enable if the agent needs to read
    // stored briefings directly. Removing it from the offered schema saves ~150 prompt tokens
    // per turn when all four pipeline tools are active.
    // @McpTool(name = "getBriefing", ...)
    public String getBriefing(
            @McpToolParam(description = "The day, as an ISO date in yyyy-MM-dd form, e.g. "
                    + "'2026-03-17'. Omit or leave empty for today.", required = false)
            String date) {

        LocalDate day;
        if (date == null || date.isBlank()) {
            day = collector.today();
        } else {
            try {
                day = LocalDate.parse(date.strip());
            } catch (DateTimeParseException e) {
                // Returned, not thrown: the model chose this string and is the only party that can
                // correct it, which it can only do if the complaint names the format.
                return "'%s' is not a date I can read. Use ISO yyyy-MM-dd, for example 2026-03-17."
                        .formatted(date);
            }
        }

        Briefing briefing = store.briefing(day);
        log.info("getBriefing({}) -> {}", day, briefing.isPresent() ? briefing.figures() : "nothing stored");

        if (!briefing.isPresent()) {
            // Said plainly, because the honest answer and the wrong answer look similar otherwise: a
            // day with no snapshot is not a day with nothing on it, and a model told "0 events" would
            // report a free day to someone who simply had the server switched off.
            return ("No briefing was collected for %s. That means nothing was recorded for that day, "
                    + "not that the day was empty — use getSchedule to read the calendar directly.")
                    .formatted(day);
        }

        StringBuilder out = new StringBuilder(320)
                .append("Briefing for ").append(day).append(" (collected ")
                .append(briefing.collectedAt()).append("):\n")
                .append(briefing.figures());
        if (briefing.narrated()) {
            out.append("\n\n").append(briefing.narrative());
        }
        if (!briefing.highlights().isEmpty()) {
            out.append("\n\nThe day:");
            briefing.highlights().forEach(line -> out.append("\n- ").append(line));
        }
        return out.toString();
    }

    // @McpTool disabled — same reason as getBriefing above.
    // @McpTool(name = "getTrend", ...)
    public String getTrend(
            @McpToolParam(description = "How many days back to include, counting today. "
                    + "1 to 14; defaults to 7.", required = false) Integer days) {

        int window = (days == null || days <= 0) ? 7 : Math.min(days, MAX_DAYS);
        LocalDate today = collector.today();
        List<Briefing> stored = store.lastDays(today, window);
        Trend trend = Briefings.trend(stored);
        log.info("getTrend({}) -> {} day(s) with a snapshot", window, trend.days());

        if (trend.isEmpty()) {
            return ("No briefings have been collected in the last %d days, so there is no trend to "
                    + "report yet.").formatted(window);
        }

        StringBuilder out = new StringBuilder(320)
                .append("Across ").append(trend.days())
                .append(trend.days() == 1 ? " collected day (" : " collected days (")
                .append(trend.from()).append(" to ").append(trend.to()).append("):\n")
                .append("- ").append(trend.events()).append(" events in total\n")
                .append("- ").append(Briefings.hoursAndMinutes(trend.bookedMinutes()))
                .append(" booked in total, ")
                .append(Briefings.hoursAndMinutes(trend.averageBookedMinutesPerDay()))
                .append(" a day on average\n")
                .append("- busiest day: ").append(trend.busiestDate()).append('\n')
                .append("- tasks overdue as of ").append(trend.to()).append(": ")
                .append(trend.tasksOverdue());

        if (trend.days() < window) {
            // The gap is reported rather than smoothed over. Days with no snapshot are absent, not
            // quiet, and an average that divided by the calendar would turn downtime into free time.
            out.append("\n\nNote: ").append(window - trend.days())
                    .append(" of the last ").append(window)
                    .append(" days have no snapshot — they were not collected, so they are excluded "
                            + "rather than counted as empty.");
        }
        return out.toString();
    }
}
