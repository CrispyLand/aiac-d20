package com.crispyland.mcpserver.notes;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.ObjectMapper;

@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties(NotesProperties.class)
public class NotesConfiguration {

    @Bean
    public NotesSummarizer notesSummarizer(ObjectMapper mapper, NotesProperties properties) {
        NotesProperties.Groq groq = properties.groq();
        if (!groq.hasKey()) {
            return NotesSummarizer.NONE;
        }
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory();
        requestFactory.setReadTimeout(groq.timeout());
        RestClient restClient = RestClient.builder().requestFactory(requestFactory).build();
        return new GroqNotesSummarizer(restClient, mapper, groq);
    }
}
