#!/usr/bin/env python3

import os
import yaml
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import config


def filter_files(root_dir, include_dirs=None, exclude_dirs=None):
    """
    Given a list of directories, yield full paths to files inside them.
    If valid_exts is a tuple/list of extensions, only yield those matching.
    """
    if include_dirs is None:
        include_dirs = []
    if exclude_dirs is None:
        exclude_dirs = []

    for subdir in include_dirs:
        include_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(include_path):
            print(f"[WARN] {include_path} is not a valid directory.")
            continue

        for root, dirs, files in os.walk(include_path):
            # Remove excluded directories from dirs in-place so os.walk doesn't descend into them.
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file_name in files:
                yield os.path.join(root, file_name)


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits `text` into overlapping chunks of size `chunk_size`.
    Each chunk overlaps the previous one by `overlap` characters.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def ingest_path(embedder, collection, file_path, chunk_count):
    print(f"Ingesting {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        return

    chunks = chunk_text(text, chunk_size=config.processing["chunk_size"], overlap=config.processing["overlap"])
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()  # convert np.array -> list
        chunk_id = f"{file_path}-{i}"  # unique ID
        metadata = {"source_file": file_path}

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[chunk_id],
            metadatas=[metadata]
        )
        chunk_count += 1


def main():
    client = chromadb.PersistentClient(path=config.chroma_persist_directory)
    collection = client.get_or_create_collection(name=config.chroma_collection_name)
    embedder = SentenceTransformer(config.embedder_model_path)

    file_count = 0
    chunk_count = 0

    for repo in config.ingest_repositories:
        for file_path in filter_files(repo["path"], include_dirs=repo["include_dirs"], exclude_dirs=repo["exclude_dirs"]):
            file_count += 1
            ingest_path(embedder, collection, file_path, chunk_count)

    print(f"[INFO] Processed {file_count} files.")
    print(f"[INFO] Added {chunk_count} chunks to collection '{CHROMA_COLLECTION}'.")
    print(f"[INFO] Chroma DB persisted at: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()

