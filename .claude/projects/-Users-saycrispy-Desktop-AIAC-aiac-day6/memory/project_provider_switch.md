---
name: LLM provider switched from Groq to Cerebras
description: Free-tier Groq replaced with Cerebras (user has $15 credits). All endpoints, key env vars, and model names updated.
type: project
---

Switched from Groq to Cerebras because Groq free tier was too restrictive for the 6-tool pipeline.

**Why:** Groq free tier hit rate/token limits during the Day 20 multi-tool morning brief chain.

**How to apply:** 
- API key env var is now `CEREBRAS_API_KEY` (was `GROQ_API_KEY`)
- Endpoint is `https://api.cerebras.ai/v1/chat/completions`
- Models: `llama-3.3-70b` (default/chat, 128k context), `llama3.1-70b` (8k), `llama3.1-8b` (background tasks)
- Cerebras does NOT support `reasoning_effort` — always `""` in all configs
- Background tasks (summarizer, extractor, compression, invariant guard, narrator, notes) use `llama3.1-8b`
- The Java class is still named `GroqLlmClient` (internal, not user-facing)
- The notes/briefing YAML section is still named `notes.groq.*` (YAML binding key, not worth renaming)
