package com.crispyland.mcpserver.google;

import com.google.api.client.util.DateTime;
import com.google.api.services.calendar.Calendar;
import com.google.api.services.calendar.model.CalendarListEntry;
import com.google.api.services.calendar.model.Event;
import com.google.api.services.calendar.model.EventDateTime;
import com.google.api.services.calendar.model.Events;
import java.io.IOException;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * A day's events, fetched and nothing more.
 * <p>
 * Split out of {@code CalendarTools} when a second consumer appeared. The tool turns events into
 * prose for a language model; the briefing job counts minutes. Neither can use the other's output,
 * but both need the same query — and the query is the part with the rule in it. Leaving the fetch
 * inside the tool would have meant the job either re-deriving the local-day bounds, or parsing
 * sentences back into times.
 * <p>
 * Read-only by construction: the {@link Calendar} arrives authorized with a readonly scope, and
 * nothing here asks it for anything else.
 */
@Service
public class CalendarReader {

    private static final Logger log = LoggerFactory.getLogger(CalendarReader.class);

    /** Enough for any single day across all calendars combined. */
    private static final int MAX_EVENTS_PER_CALENDAR = 50;

    private final Calendar calendar;
    private final ZoneId zone;

    public CalendarReader(Calendar calendar, GoogleProperties properties) {
        this.calendar = calendar;
        this.zone = properties.zone();
    }

    /** The zone these events were placed in, for anything that has to read their clock times. */
    public ZoneId zone() {
        return zone;
    }

    /**
     * Everything on the given local day across ALL of the user's calendars.
     * <p>
     * The query uses a wide UTC window (day−1 to day+2) rather than timezone-adjusted boundaries.
     * All-day events are stored as plain dates in the calendar's own timezone, which may differ
     * from the server's configured zone — a Shanghai-shifted window would silently miss events
     * created in a UTC or western-timezone account. The wide window catches them all, and the
     * results are then filtered by matching the event's stored date (or dateTime-derived date in
     * the server's zone) against the requested day.
     * <p>
     * Never null — an empty list is the honest answer for a free day.
     */
    public List<Event> eventsOn(LocalDate day) throws IOException {
        // Widen the UTC window by ±1 day to catch all-day events in any calendar timezone.
        // An all-day event on `day` in UTC-12 starts at `day+12h UTC`; one in UTC+14 starts at
        // `day-14h UTC`. A ±1 day buffer around the requested date covers the full range.
        DateTime timeMin = new DateTime(day.minusDays(1).atStartOfDay(java.time.ZoneOffset.UTC)
                .toInstant().toEpochMilli());
        DateTime timeMax = new DateTime(day.plusDays(2).atStartOfDay(java.time.ZoneOffset.UTC)
                .toInstant().toEpochMilli());

        List<CalendarListEntry> calendars = calendar.calendarList().list().execute().getItems();
        if (calendars == null || calendars.isEmpty()) {
            log.info("eventsOn({}): no calendars found in list", day);
            return List.of();
        }

        List<Event> all = new ArrayList<>();
        for (CalendarListEntry cal : calendars) {
            String id = cal.getId();
            try {
                Events result = calendar.events().list(id)
                        .setTimeMin(timeMin)
                        .setTimeMax(timeMax)
                        .setSingleEvents(true)
                        .setOrderBy("startTime")
                        .setMaxResults(MAX_EVENTS_PER_CALENDAR)
                        .execute();
                List<Event> items = result.getItems();
                if (items != null) {
                    for (Event e : items) {
                        if (fallsOn(e, day)) {
                            all.add(e);
                        }
                    }
                }
            } catch (IOException e) {
                log.warn("eventsOn({}): could not read calendar '{}': {}", day, id, e.getMessage());
            }
        }

        log.info("eventsOn({}) -> {} event(s) across {} calendar(s)", day, all.size(), calendars.size());
        return all;
    }

    /**
     * Whether an event falls on the requested day.
     * <p>
     * All-day events carry a plain {@code date} string ("2026-09-28") — match it directly without
     * any timezone conversion, because the date is already in the calendar's own timezone and is
     * the canonical answer to "which day is this on?".
     * <p>
     * Timed events carry a {@code dateTime} timestamp — convert it to the server's configured zone
     * to decide which local day it lands in, the same logic as before.
     */
    private boolean fallsOn(Event event, LocalDate day) {
        EventDateTime start = event.getStart();
        if (start == null) {
            return false;
        }
        if (start.getDate() != null) {
            // All-day event: the date string is authoritative regardless of timezone.
            try {
                return LocalDate.parse(start.getDate().toStringRfc3339()).equals(day);
            } catch (Exception e) {
                return false;
            }
        }
        if (start.getDateTime() != null) {
            // Timed event: convert timestamp to the server's local zone.
            LocalDate eventDay = Instant.ofEpochMilli(start.getDateTime().getValue())
                    .atZone(zone).toLocalDate();
            return eventDay.equals(day);
        }
        return false;
    }
}
