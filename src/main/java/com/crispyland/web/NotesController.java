package com.crispyland.web;

import com.crispyland.agent.AgentProperties;
import java.time.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.client.RestClient;

/**
 * Proxies file downloads from the MCP server's notes directory to the browser.
 * <p>
 * The MCP server binds on {@code 127.0.0.1:8081} — it is not reachable from the internet
 * directly. This endpoint fetches the file on the browser's behalf and streams it back as an
 * attachment, so the browser sees a clean download from the agent's own port (80 via nginx)
 * without needing any extra firewall rule.
 */
@Controller
public class NotesController {

    private static final Logger log = LoggerFactory.getLogger(NotesController.class);

    private final RestClient restClient;
    private final String mcpBaseUrl;

    public NotesController(AgentProperties properties) {
        String baseUrl = properties.briefing().url();
        this.mcpBaseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory();
        factory.setReadTimeout(Duration.ofSeconds(10));
        this.restClient = RestClient.builder().requestFactory(factory).build();
    }

    @GetMapping("/notes/download")
    public ResponseEntity<byte[]> download(@RequestParam String file) {
        // Strip everything the MCP server would also strip — the proxy should never forward
        // a filename it knows will be rejected or could escape the notes directory.
        String safe = file.replaceAll("[^A-Za-z0-9.\\-_]", "").replaceAll("^\\.+", "");
        if (safe.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        try {
            byte[] body = restClient.get()
                    .uri(mcpBaseUrl + "/notes/download?file=" + safe)
                    .retrieve()
                    .onStatus(status -> !status.is2xxSuccessful(), (req, res) -> { })
                    .body(byte[].class);
            if (body == null || body.length == 0) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
            }
            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + safe + "\"")
                    .header(HttpHeaders.CONTENT_TYPE, "application/octet-stream")
                    .body(body);
        } catch (Exception e) {
            log.warn("Notes download failed for '{}': {}", safe, e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).build();
        }
    }
}
