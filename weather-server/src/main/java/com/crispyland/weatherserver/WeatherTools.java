package com.crispyland.weatherserver;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.mcp.annotation.McpTool;
import org.springframework.ai.mcp.annotation.McpToolParam;
import org.springframework.stereotype.Service;

/**
 * One MCP tool: today's weather for a city, backed by Open-Meteo (no API key required).
 * <p>
 * The result is prose: current temperature, feels-like, today's high and low, precipitation
 * probability and amount, wind speed, and an outdoor advisory. The model reads this as a string,
 * so prose is the right format — it avoids JSON punctuation and token overhead while keeping
 * everything the model needs for a morning brief.
 * <p>
 * The server is read-only by construction: Open-Meteo has no write endpoints, so there is nothing
 * for this tool to accidentally modify.
 */
@Service
public class WeatherTools {

    private static final Logger log = LoggerFactory.getLogger(WeatherTools.class);

    private final OpenMeteoClient client;

    public WeatherTools(OpenMeteoClient client) {
        this.client = client;
    }

    @McpTool(name = "getWeather",
            description = "Get today's weather for a city: current temperature, feels-like, "
                    + "today's high and low, precipitation probability and amount, wind speed, "
                    + "and an outdoor advisory. City defaults to Shanghai when omitted. "
                    + "Read-only: backed by Open-Meteo, no API key required.")
    public String getWeather(
            @McpToolParam(description = "City name, e.g. 'Shanghai' or 'London'. "
                    + "Omit to use the server's default city.", required = false) String city) {
        String result = client.report(city);
        log.info("getWeather({}) -> {}", city, result.lines().findFirst().orElse(""));
        return result;
    }
}
