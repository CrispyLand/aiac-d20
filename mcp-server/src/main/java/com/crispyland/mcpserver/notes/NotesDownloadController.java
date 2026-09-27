package com.crispyland.mcpserver.notes;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Serves files from the notes directory as downloads. Only the bare filename is accepted —
 * path separators are stripped by {@link NotesTools#sanitize}, so it is impossible to walk
 * above the notes directory regardless of what the caller supplies.
 * <p>
 * Bound on {@code 127.0.0.1} in production, so only the agent on the same host can reach it.
 * The agent proxies it to the browser; the browser never talks to this port directly.
 */
@RestController
public class NotesDownloadController {

    private final Path notesDir;

    public NotesDownloadController(NotesProperties properties) {
        this.notesDir = properties.path();
    }

    @GetMapping("/notes/download")
    public ResponseEntity<Resource> download(@RequestParam String file) throws IOException {
        String safe = NotesTools.sanitize(file);
        if (safe.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        Path target = notesDir.resolve(safe);
        if (!Files.exists(target) || !Files.isRegularFile(target)) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + safe + "\"")
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .contentLength(Files.size(target))
                .body(new FileSystemResource(target));
    }
}
