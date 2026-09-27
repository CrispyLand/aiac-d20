package com.crispyland.weatherserver;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties(WeatherProperties.class)
public class WeatherServerApplication {

    public static void main(String[] args) {
        SpringApplication.run(WeatherServerApplication.class, args);
    }
}
