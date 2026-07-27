#!/usr/bin/env python3
"""
Experiment 1: Per-Model Carbon Benchmark
=========================================
Runs each of GreenServ's 16 models on 10 general-knowledge queries and
measures carbon emissions using CodeCarbon's OfflineEmissionsTracker.

Establishes baseline carbon per model to understand how parameter count
and model architecture affect carbon footprint.

Hardware: NVIDIA L40S GPU on Laguna HPC (USC/CARC)
Carbon measurement: OfflineEmissionsTracker, country_iso_code="USA"

Results saved to: carbon_results.csv (raw CodeCarbon output)
                  carbon_summary.csv (per-model summary)

Usage:
    python3 carbon_benchmark.py

Note: Gated models (Gemma, Llama) require HuggingFace token and
      accepted terms of use at huggingface.co.
      Set HF_TOKEN environment variable before running.
"""

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import OfflineEmissionsTracker

# GreenServ's 16 model pool with HuggingFace identifiers
MODELS = {
    "qwen2.5-0.5b":  "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b":  "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":    "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b":   "Qwen/Qwen2.5-14B-Instruct",
    "mistral-7b":    "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-3-1b":    "google/gemma-3-1b-it",       # requires HF token + terms
    "gemma-3-4b":    "google/gemma-3-4b-it",       # requires HF token + terms
    "gemma-3-12b":   "google/gemma-3-12b-it",      # requires HF token + terms
    "gemma-3-27b":   "google/gemma-3-27b-it",      # requires HF token + terms
    "llama-3.1-8b":  "meta-llama/Llama-3.1-8B-Instruct",  # requires HF token + terms
    "llama-3.2-1b":  "meta-llama/Llama-3.2-1B-Instruct",  # requires HF token + terms
    "llama-3.2-3b":  "meta-llama/Llama-3.2-3B-Instruct",  # requires HF token + terms
    "phi-4-mini-4b": "microsoft/Phi-4-mini-instruct",
    "phi-4-14b":     "microsoft/phi-4",
    "yi-34b":        "01-ai/Yi-1.5-34B-Chat",
}

# 10 general-knowledge test queries
QUERIES = [
    "What is the capital of France?",
    "Solve: 2 + 2 = ?",
    "Summarize: The sun is a star.",
    "What is machine learning?",
    "Translate to Spanish: Hello world.",
    "What is the boiling point of water?",
    "Who wrote Romeo and Juliet?",
    "What is photosynthesis?",
    "What is the speed of light?",
    "Name the planets in the solar system.",
]

OUTPUT_DIR = "/project/JehoPark_1895/Yuna_SLM"  # update for your environment

results = []

for model_id, model_name in MODELS.items():
    print(f"\nRunning {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        tracker = OfflineEmissionsTracker(
            project_name=model_id,
            country_iso_code="USA",
            log_level="error",
            save_to_file=True,
            output_file=f"{OUTPUT_DIR}/carbon_results.csv"
        )
        tracker.start()

        for query in QUERIES:
            inputs = tokenizer(query, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, max_new_tokens=50)

        emissions_kg = tracker.stop()
        emissions_g = emissions_kg * 1000

        results.append({
            "model_id":          model_id,
            "model_name":        model_name,
            "carbon_emissions_g": emissions_g,
            "queries_run":       len(QUERIES)
        })
        print(f"  {model_id}: {emissions_g:.4f} gCO2")

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR on {model_id}: {e}")
        results.append({
            "model_id":          model_id,
            "model_name":        model_name,
            "carbon_emissions_g": None,
            "queries_run":       0
        })

df = pd.DataFrame(results)
df.to_csv(f"{OUTPUT_DIR}/carbon_summary.csv", index=False)
print("\nDone!")
print(df[["model_id", "carbon_emissions_g"]].to_string())
