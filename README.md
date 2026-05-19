# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

Name: Anan Madhav T V

Roll No: MM22B013

---
## Repository

GitHub Repository:

[https://github.com/ananmadhav/da6401-assignment-3]

---

## W&B Report

Weights & Biases Report:

[https://api.wandb.ai/links/mm22b013-/p5esm2gg]

---

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

--

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
---

## Project Overview
This repository contains the implementation and experimental analysis for DA6401 Assignment 3. The project investigates several architectural and training choices in Transformer models through systematic ablation studies and visualization using Weights & Biases (W&B).

The experiments include:

- **2.1 The Necessity of the Noam Scheduler**
- **2.2 Ablation: Scaling Factor in Attention**
- **2.3 Attention Rollout & Head Specialization**
- **2.4 Positional Encoding vs Learned Embeddings**
- **2.5 Decoder Sensitivity: Label Smoothing**

---

## Experiments Conducted

### 2.1 The Necessity of the Noam Scheduler
Compared:
- Noam Scheduler
- Fixed Learning Rate

Metrics analyzed:
- Training Loss
- Validation Accuracy

---

### 2.2 Ablation: Scaling Factor

Compared:
- Scaled Attention
- Unscaled Attention

Metrics analyzed:
- Query Gradient Norm
- Key Gradient Norm
- Training Loss
- Validation Accuracy

---

### 2.3 Attention Rollout & Head Specialization

Visualized:
- Attention heatmaps for all encoder heads
- Head specialization behavior
- Head redundancy analysis

---

### 2.4 Positional Encoding vs Learned Embeddings

Compared:
- Sinusoidal Positional Encoding
- Learned Positional Embedding

Metrics analyzed:
- Validation BLEU
- Validation Loss

---

### 2.5 Decoder Sensitivity: Label Smoothing

Compared:
- Label Smoothing (ε = 0.1)
- Standard Cross Entropy (ε = 0.0)

Metrics analyzed:
- Prediction Confidence
- Perplexity

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py
```





