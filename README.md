# Sult

con/sult is an LLM-driven RAG chatbot that consumes configurable repositories of context.

## Ingest Repositories

(Current config consumes versations-archive)

`python -m jobs.directory_ingestor`

## Demo Chatbot

`python chatbot/main.py`

    User: How do I specify output prefix with duct?
    Assistant:
