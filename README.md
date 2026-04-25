# Toxicity Classification in Gaming Chat

**CS 490: Natural Language Processing — Spring 2026, Purdue University**

Kylie Houston, Suhaas Nachannagari, Rishi Shekhar, Josh Rubow

## Overview

This project fine-tunes transformer-based models to classify in-game chat messages into three categories: **Neutral**, **Banter**, and **Toxic**. We compare four model architectures to evaluate how contextual embeddings and parameter-efficient fine-tuning improve classification over a statistical baseline.

## Models

| Script | Model | Description |
|---|---|---|
| `nbc.py` | Naive Bayes | TF-IDF + Multinomial Naive Bayes baseline |
| `bert_baseline.py` | BERT-base | Full fine-tuning with a classification head |
| `bert_lora.py` | BERT-base + LoRA | Parameter-efficient fine-tuning via LoRA |
| `bert_large.py` | BERT-large + LoRA | Larger backbone with LoRA fine-tuning |

## Project Structure

```
├── setup.sh              # Environment setup script
├── requirements.txt      # Python dependencies
├── README.md
├── train.csv             # Training split
├── val.csv               # Validation split
├── test.csv              # Test split
├── nbc.py                # Naive Bayes classifier
├── bert_baseline.py      # BERT-base full fine-tuning
├── bert_lora.py          # BERT-base + LoRA
└── bert_large.py         # BERT-large + LoRA
```

## Setup

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

Or install manually:

```bash
pip install -r requirements.txt
```

## Running the Models

Each script loads data from the `data/` directory, trains the model, and prints the classification report and confusion matrix to the console.

```bash
python nbc.py
python bert_baseline.py
python bert_lora.py
python bert_large.py
```

**Note:** The BERT models require a CUDA-capable GPU for reasonable training times. They were developed and tested on Google Colab T4 GPU with converging times aroudn 8 minutes for each model.

## Dataset

The dataset is derived from the [DOTA 2 Toxic Chat Dataset](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) on Hugging Face. Messages were re-annotated into three classes (Neutral, Banter, Toxic) manually.