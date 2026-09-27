package com.crispyland.weatherserver;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Two Open-Meteo calls: geocoding (city name → lat/lon) then forecast.
 * <p>
 * Open-Meteo is free and requires no API key — the right choice for a demo server that has to
 * just work without sign-up. Both endpoints are read-only GET requests over HTTPS.
 */
@Component
public class OpenMeteoClient {

    private static final Logger log = LoggerFactory.getLogger(OpenMeteoClient.class);

    private static final String GEOCODING_URL =
            "https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json";

    private static final String FORECAST_URL =
            "https://api.open-meteo.com/v1/forecast"
            + "?latitude={lat}&longitude={lon}"
            + "&current=temperature_2m,apparent_temperature,weathercode,precipitation,windspeed_10m"
            + "&daily=weathercode,temperature_2m_max,temperature_2m_min"
            +        ",precipitation_sum,precipitation_probability_max"
            + "&timezone={tz}&forecast_days=1";

    private final RestClient http;
    private final ObjectMapper mapper;
    private final WeatherProperties properties;

    public OpenMeteoClient(RestClient http, ObjectMapper mapper, WeatherProperties properties) {
        this.http = http;
        this.mapper = mapper;
        this.properties = properties;
    }

    /**
     * Full weather report for {@code city} today. Returns a human-readable string the model can
     * include directly in a morning brief, including an explicit outdoor advisory line.
     */
    public String report(String city) {
        String target = (city == null || city.isBlank()) ? properties.defaultCity() : city.strip();

        double[] coords = geocode(target);
        if (coords == null) {
            return "Could not find '%s' in the geocoding database. Check the city name.".formatted(target);
        }

        String json = http.get()
                .uri(FORECAST_URL, coords[0], coords[1], properties.defaultTimezone())
                .retrieve()
                .body(String.class);

        return parse(target, json);
    }

    private double[] geocode(String city) {
        try {
            String json = http.get()
                    .uri(GEOCODING_URL, city)
                    .retrieve()
                    .body(String.class);
            JsonNode results = mapper.readTree(json).path("results");
            if (results.isMissingNode() || results.isEmpty()) {
                return null;
            }
            JsonNode first = results.get(0);
            return new double[]{first.path("latitude").doubleValue(),
                                 first.path("longitude").doubleValue()};
        } catch (Exception e) {
            log.warn("Geocoding '{}' failed: {}", city, e.getMessage());
            return null;
        }
    }

    private String parse(String city, String json) {
        try {
            JsonNode root = mapper.readTree(json);
            JsonNode current = root.path("current");
            JsonNode daily = root.path("daily");

            double tempNow = current.path("temperature_2m").doubleValue();
            double feelsLike = current.path("apparent_temperature").doubleValue();
            int currentCode = current.path("weathercode").intValue();
            double precipNow = current.path("precipitation").doubleValue();
            double wind = current.path("windspeed_10m").doubleValue();

            int dailyCode = daily.path("weathercode").path(0).intValue();
            double tempMax = daily.path("temperature_2m_max").path(0).doubleValue();
            double tempMin = daily.path("temperature_2m_min").path(0).doubleValue();
            double precipSum = daily.path("precipitation_sum").path(0).doubleValue();
            int precipProb = daily.path("precipitation_probability_max").path(0).intValue();

            String date = LocalDate.now().format(DateTimeFormatter.ISO_LOCAL_DATE);

            log.info("getWeather({}) → {}°C, code={}, precip={}mm, prob={}%",
                    city, tempNow, dailyCode, precipSum, precipProb);

            return """
                    Weather in %s on %s:
                      Current: %.0f°C (feels like %.0f°C), %s
                      Today: High %.0f°C / Low %.0f°C
                      Rain: %.1f mm expected (%d%% probability)
                      Wind: %.0f km/h

                    Outdoor advisory: %s"""
                    .formatted(city, date,
                            tempNow, feelsLike, describe(currentCode),
                            tempMax, tempMin,
                            precipSum, precipProb,
                            wind,
                            advisory(dailyCode, precipSum, precipProb));
        } catch (Exception e) {
            log.warn("Parsing Open-Meteo response failed: {}", e.getMessage());
            return "Weather data could not be parsed. The service may be temporarily unavailable.";
        }
    }

    // ── WMO weather interpretation codes ──────────────────────────────────────

    static String describe(int code) {
        return switch (code) {
            case 0 -> "Clear sky";
            case 1 -> "Mainly clear";
            case 2 -> "Partly cloudy";
            case 3 -> "Overcast";
            case 45, 48 -> "Fog";
            case 51, 53 -> "Drizzle";
            case 55 -> "Heavy drizzle";
            case 56, 57 -> "Freezing drizzle";
            case 61 -> "Light rain";
            case 63 -> "Moderate rain";
            case 65 -> "Heavy rain";
            case 66, 67 -> "Freezing rain";
            case 71 -> "Light snow";
            case 73 -> "Moderate snow";
            case 75 -> "Heavy snow";
            case 77 -> "Snow grains";
            case 80 -> "Light rain showers";
            case 81 -> "Moderate rain showers";
            case 82 -> "Violent rain showers";
            case 85, 86 -> "Snow showers";
            case 95 -> "Thunderstorm";
            case 96, 99 -> "Thunderstorm with hail";
            default -> "Conditions unknown";
        };
    }

    static String advisory(int dailyCode, double precipSum, int precipProb) {
        if (dailyCode == 95 || dailyCode == 96 || dailyCode == 99) {
            return "Thunderstorm expected — cancel all outdoor activities.";
        }
        if ((dailyCode >= 71 && dailyCode <= 77) || dailyCode == 85 || dailyCode == 86) {
            return "Snow expected — dress warmly, outdoor activities may be hazardous.";
        }
        if (precipSum >= 5 || precipProb >= 70) {
            return "Heavy rain expected — bring an umbrella or reschedule outdoor activities.";
        }
        if (dailyCode >= 61 || precipSum >= 2 || precipProb >= 50) {
            return "Rain likely — bring an umbrella. Consider moving outdoor activities indoors.";
        }
        if (dailyCode >= 51 || precipProb >= 30) {
            return "Light drizzle possible — an umbrella is handy just in case.";
        }
        return "Good conditions for outdoor activities.";
    }
}
