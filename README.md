# Bigram Language Model

## Overview
This project implements a **Bigram Model** trained on classic literary works such as *Frankenstein's Monster*, *Alice's Adventures in Wonderland*, and *Dr. Jekyll and Mr. Hyde*. The model learns probabilistic word transitions to generate text with a style reminiscent of these books.

## Features
- Trains a bigram language model on selected texts
- Generates sequences based on the training corpus
- Provides insight into word pair frequency distributions

## Data Sources
The model is trained on:
- Frankenstein by Mary Shelley
- Alice’s Adventures in Wonderland by Lewis Carroll
- Dr. Jekyll and Mr. Hyde by Robert Louis Stevenson
- Moby Dick; Or, The Whale by Herman Melville
- The Wonderful Wizard of Oz by L. Frank Baum

## Tech Stack
### Languages:
Python

### Libraries:
PyTorch
NumPy
Matplotlib (for loss visualization)

### Tools:
Jupyter Notebook / VS Code
Git

## Installation
### Clone the repository
- git clone https://github.com/Sriramsurya02/Language-Model.git
- cd Language-Model

### (Optional) create a cuda virtual environment
python -m venv cuda
source cuda/bin/activate  # on Windows: cuda\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Add the required Training data (text from books) to books.txt

### Run the Model
python LLM1.py

## Results
- Model converges quickly on small datasets
- Generates stylistically consistent text with recognizable sentence flow
- Demonstrates how simple statistical language models capture word dependencies

### References
## Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"
## PyTorch Official Documentation: https://pytorch.org/docs/stable/
## Datasets: Public domain texts from Project Gutenberg
