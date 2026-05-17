#!/usr/bin/env python3
"""
🚀 IBM AI Engineering — Smart Sync Script
─────────────────────────────────────────
Automatically detects the right chapter folder, moves the notebook,
and pushes to GitHub in one command.

Usage:
    python scripts/sync_notebook.py <path/to/notebook.ipynb>
    python scripts/sync_notebook.py <path/to/notebook.ipynb> "custom commit message"

Examples:
    python scripts/sync_notebook.py ~/Downloads/ANN_Forward_Propagation.ipynb
    python scripts/sync_notebook.py ~/Downloads/CNN_Basics.ipynb "📷 CNN basics done"
"""

import sys
import os
import shutil
import subprocess
import json
from datetime import datetime
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
BRANCH    = "main"

# Chapter keyword map — updated to match real folder names (May 2026)
CHAPTER_MAP = {
    "01_ML_with_Python": [
        "supervised", "unsupervised", "regression", "classification",
        "clustering", "svm", "random forest", "decision tree",
        "pipeline", "gridsearch", "scikit", "sklearn", "knn",
        "pca", "tsne", "umap", "recommender", "collaborative filtering"
    ],
    "02_Intro_Deep_Learning_Neural_Networks_Keras": [
        "keras", "neural network", "ann", "forward propagation",
        "backpropagation", "deep learning", "activation", "sigmoid",
        "relu", "dense", "sequential", "epoch", "batch",
        "intro to deep", "introduction to deep"
    ],
    "03_Deep_Learning_Keras_TensorFlow": [
        "tensorflow", "tf.", "custom layer", "gan", "autoencoder",
        "generative adversarial", "time series", "conv2d", "keras tensorflow",
        "deep q", "dqn", "reinforcement keras"
    ],
    "04_Intro_Neural_Networks_PyTorch": [
        "pytorch", "torch", "tensor", "autograd", "nn.module",
        "dataloader", "intro pytorch", "introduction pytorch",
        "linear regression pytorch", "logistic regression pytorch"
    ],
    "05_Deep_Learning_PyTorch": [
        "deep pytorch", "cnn pytorch", "convolutional pytorch",
        "dropout", "batch normalization", "batchnorm", "weight init",
        "softmax pytorch", "transfer learning pytorch"
    ],
    "06_AI_Capstone_Deep_Learning": [
        "capstone", "end-to-end", "geospatial", "land classification",
        "vision transformer", "vit", "image classification project"
    ],
    "07_GenAI_LLMs_Architecture_Data_Prep": [
        "generative ai", "llm", "large language", "tokenization",
        "tokenizer", "nltk", "spacy", "bertokenizer", "xlnet",
        "diffusion", "vae", "variational autoencoder", "architecture llm"
    ],
    "08_GenAI_Foundational_Models_NLP": [
        "word2vec", "cbow", "skip-gram", "embedding", "n-gram",
        "seq2seq", "encoder decoder", "rnn nlp", "foundational nlp",
        "one-hot", "bag of words", "bow"
    ],
    "09_GenAI_Language_Modeling_Transformers": [
        "attention mechanism", "self-attention", "positional encoding",
        "gpt", "bert", "transformer model", "hugging face", "masked",
        "causal language", "text classification transformer"
    ],
    "10_GenAI_Engineering_Fine_Tuning_Transformers": [
        "fine-tuning", "fine tuning", "lora", "qlora", "peft",
        "instruction tuning", "sft", "supervised fine", "hugging face fine"
    ],
    "11_GenAI_Advanced_Fine_Tuning_LLMs": [
        "rlhf", "dpo", "direct preference", "reward model",
        "ppo", "proximal policy", "alignment", "advanced fine tuning"
    ],
    "12_AI_Agents_RAG_LangChain": [
        "rag", "langchain", "agent", "retrieval", "vector store",
        "chroma", "faiss", "tool use", "chain", "retrieval augmented",
        "prompt engineering", "in-context learning"
    ],
    "13_Project_GenAI_Apps_RAG_LangChain": [
        "gradio", "qa bot", "question answering bot", "final project genai",
        "rag app", "vector database project", "langchain project"
    ],
}
# ──────────────────────────────────────────────────────────────────────────────


def run(cmd, check=True):
    """Run a shell command and print output."""
    print(f"  ▶ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0 and check:
        print(f"  ❌ Error: {result.stderr.strip()}")
        sys.exit(1)
    return result


def validate_notebook(path):
    """Validate that the file is a proper Jupyter notebook."""
    try:
        with open(path) as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        print(f"  ✅ Valid notebook — {len(cells)} cells")
        return nb
    except Exception as e:
        print(f"  ❌ Invalid notebook: {e}")
        sys.exit(1)


def extract_notebook_text(nb):
    """Extract all text content from notebook cells for keyword matching."""
    text = ""
    for cell in nb.get("cells", []):
        for line in cell.get("source", []):
            text += line.lower() + " "
    return text


def detect_chapter(nb, filename):
    """Auto-detect the chapter folder based on notebook content and filename."""
    text = extract_notebook_text(nb) + filename.lower()

    scores = {}
    for chapter, keywords in CHAPTER_MAP.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[chapter] = score

    if scores:
        best = max(scores, key=scores.get)
        print(f"  🎯 Auto-detected chapter: {best} (score: {scores[best]})")
        return best

    # Fallback — ask user
    print("\n  ⚠️  Could not auto-detect chapter. Please choose:")
    chapters = list(CHAPTER_MAP.keys())
    for i, ch in enumerate(chapters, 1):
        print(f"    {i:>2}. {ch}")
    choice = input("\n  Enter number: ").strip()
    try:
        return chapters[int(choice) - 1]
    except:
        print("  ❌ Invalid choice")
        sys.exit(1)


def generate_commit_message(nb, filename):
    """Generate a smart commit message from the notebook title."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            for line in cell.get("source", []):
                if line.startswith("# "):
                    title = line.strip("# ").strip()
                    return f"🧠 Add: {title}"
    name = Path(filename).stem.replace("_", " ")
    return f"🧠 Add: {name}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    nb_path = Path(sys.argv[1]).expanduser().resolve()
    custom_msg = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n🚀 IBM AI Engineering — Smart Sync\n{'─'*40}")
    print(f"📎 Notebook: {nb_path.name}\n")

    # 1. Validate
    print("📋 Step 1 · Validating notebook...")
    nb = validate_notebook(nb_path)

    # 2. Detect chapter
    print("\n🔍 Step 2 · Detecting chapter...")
    chapter = detect_chapter(nb, nb_path.name)

    # 3. Copy to chapter folder
    dest_dir = REPO_ROOT / chapter
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / nb_path.name
    print(f"\n📁 Step 3 · Copying to {chapter}/...")
    shutil.copy2(nb_path, dest)
    print(f"  ✅ Saved → {dest}")

    # 4. Jupytext pairing
    print("\n🔁 Step 4 · Jupytext pairing...")
    result = run(f"jupytext --to py:percent '{dest}'", check=False)
    if result.returncode == 0:
        print("  ✅ .py mirror created")
    else:
        print("  ⚠️  Jupytext not found — run: pip install jupytext")

    # 5. Git push
    print("\n📤 Step 5 · Pushing to GitHub...")
    msg = custom_msg or generate_commit_message(nb, nb_path.name)
    run("git add .")
    run(f'git commit -m "{msg}"')
    run(f"git push origin {BRANCH}")

    print(f"\n{'─'*40}")
    print(f"✅ Done! Pushed to GitHub → {chapter}/{nb_path.name}")
    print(f"🌐 https://github.com/IsReda/IBM-AI-Engineering\n")


if __name__ == "__main__":
    main()
