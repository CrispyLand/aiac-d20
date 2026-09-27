package com.crispyland.weatherserver;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

/**
 * Provides the HTTP client used by {@link OpenMeteoClient}.
 * <p>
 * {@code spring-ai-starter-mcp-server-webmvc} does not register a {@link RestClient.Builder}
 * bean, so we build a plain client directly. Open-Meteo requires no auth headers or base URL,
 * so nothing beyond {@code RestClient.create()} is needed here.
 */
@Configuration(proxyBeanMethods = false)
public class WeatherConfiguration {

    @Bean
    public RestClient restClient() {
        return RestClient.create();
    }
}
