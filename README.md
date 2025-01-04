Abstract-to-Title Generation using GPT-2
This repository contains a project that fine-tunes the GPT-2 language model for generating research paper titles from abstracts in the field of computer science. The model leverages Hugging Face's Transformers library and is trained on a dataset of computer science papers from arXiv.

Features
Fine-tunes the GPT-2 model on a custom dataset.
Implements mixed precision training for efficiency.
Supports gradient accumulation for handling smaller batch sizes.
Includes evaluation metrics and visualization of training loss.
Provides an interactive web app for generating titles from abstracts.
Try It Out
Visit the live app: Paper Title Generator

Dataset
The dataset consists of a CSV file (arxiv_cs_papers.csv) with the following columns:

abstract: The abstract of the paper.
title: The title of the paper.
Future Work
Experiment with larger models such as GPT-3.
Add support for multi-task learning.
Improve the evaluation metrics for title generation.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Hugging Face for the Transformers library.
OpenAI for the GPT-2 model.
