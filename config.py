import os
import yaml

def load_config(config_file="config.yaml"):
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = load_config()

# Extract sections for convenience.
ingest_repositories = _cfg.get("ingest_repositories", [])
embedder_model_path = _cfg.get("embedder_model_path")
llm = _cfg.get("llm", {})
processing = _cfg.get("processing", {})
chroma = _cfg.get("chroma", {})

# Shortcut variables (you can name them as you like)
chroma_collection_name = chroma.get("collection_name")
chroma_persist_directory = chroma.get("persist_directory")
chroma_embedding_dim = chroma.get("embedding_dim")
chroma_index_type = chroma.get("index_type")
chroma_top_k = chroma.get("top_k")

llm_model_path = llm.get("model_path")
llm_system_prompt = llm.get("system_prompt")
llm_assistant_prompt = llm.get("assistant_prompt")
