#!/usr/bin/env python3

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

import config

client = chromadb.PersistentClient(config.chroma.persist_directory)
collection = client.get_collection(name=config.chroma.collection_name)
embedder = SentenceTransformer(config.embedder_model)
tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

def retrieve_and_answer(user_prompt):
    """
    Query context db and generate response from the LLM.
    """
    # Embed the user prompt with the SentenceTransformer.
    query_embedding = embedder.encode(user_prompt).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config.chroma.top_k
    )

    # 'results' is a dict with keys: 'ids', 'embeddings', 'documents', 'metadatas'
    # We only need the text from 'documents'
    docs = results["documents"][0]  # top_k docs for this single query
    context = "\n".join(docs)
    final_prompt = f"{config.llm.system_prompt}\n\nContext:\n{context}\n\nUser: {user_prompt}\n{config.llm.assistant_prompt}"

    inputs = tokenizer(final_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if not user_input.strip():
            print("Goodbye!")
            break

        response = retrieve_and_answer(user_input)
        print("Assistant:", response)

