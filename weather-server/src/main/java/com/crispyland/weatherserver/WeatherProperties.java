package com.crispyland.weatherserver;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Where to look by default and how to interpret a day boundary.
 *
 * @param defaultCity     city used when the caller omits the {@code city} parameter
 * @param defaultTimezone IANA timezone for the forecast day boundary; should match the timezone
 *                        where the user reads the morning brief
 */
@ConfigurationProperties(prefix = "weather")
public record WeatherProperties(String defaultCity, String defaultTimezone) {

    public WeatherProperties {
        defaultCity = (defaultCity == null || defaultCity.isBlank()) ? "Shanghai" : defaultCity.strip();
        defaultTimezone = (defaultTimezone == null || defaultTimezone.isBlank())
                ? "Asia/Shanghai" : defaultTimezone.strip();
    }
}
