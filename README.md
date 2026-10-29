# Transformer-Based Next-Token Prediction

A Transformer model that predicts the next possible word using causal self-attention.

## Overview

This project uses the WikiText-2 dataset to train a Transformer model for next-token prediction. It learns text patterns using token embeddings, positional embeddings, and multi-head self-attention.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Hugging Face Datasets

## Dataset

- **Dataset:** WikiText-2 Raw
- **Source:** https://huggingface.co/datasets/Salesforce/wikitext
- **Configuration:** `wikitext-2-raw-v1`

The dataset is downloaded automatically when the program runs.

## Model Architecture

1. Token Embedding
2. Positional Embedding
3. Multi-Head Self-Attention
4. Residual Connections and Layer Normalization
5. Feed-Forward Network
6. Output Layer for Next-Token Prediction

## How It Works

1. Load and tokenize the dataset.
2. Create a vocabulary and convert words into numerical IDs.
3. Prepare input and target sequences.
4. Train the Transformer model using causal self-attention.
5. Predict the top five possible next words.
6. Display the training and validation accuracy graph.

## Installation

Install the required libraries:

```bash
pip install tensorflow datasets numpy matplotlib
