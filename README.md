Legal Judgment Summarizer using Unsupervised Reinforcement Learning

A domain-specific, unsupervised abstractive/extractive summarization system designed specifically for Indian Legal Judgments. This project utilizes Deep Reinforcement Learning (Policy Gradient) to generate highly structured, multi-aspect summaries (Facts, Analysis, Arguments, Judgment, Statute), overcoming the critical bottleneck of requiring ground-truth reference summaries during the training phase.

🚀 Project Overview

Legal documents, particularly Indian court judgments, are notoriously long, complex, and unstructured. They often contain archaic terminology, intricate sentence structures, and non-linear narratives that make manual summarization a time-consuming and expensive task. Traditional supervised summarization models fail in this domain because they require massive datasets of paired (document, summary) examples, which are scarce and costly to create in the legal field.

This project solves that data scarcity bottleneck using an Unsupervised RL Agent. Instead of mimicking human summaries, the agent learns to identify and select the most salient sentences by continuously optimizing a Reference-Free Reward Function. This function is composed of intrinsic quality metrics—such as coherence, coverage, and diversity—allowing the model to "teach itself" what a good summary looks like without ever seeing one.

Key Features

Domain-Specific Encoder: Leveraging law-ai/InLegalBERT (frozen) to generate 768-dimensional embeddings. Unlike generic BERT models, this encoder is pre-trained on Indian legal texts, allowing it to capture nuances like "petitioner," "respondent," and specific acts with greater semantic accuracy.

Multi-Aspect Summarization: The model goes beyond generic summaries by segmenting the judgment into 5 distinct, professionally relevant aspects:

Facts: The procedural history and background story leading to the dispute.

Analysis: The court's logical reasoning and application of legal principles.

Argument: The specific contentions and counter-claims presented by the petitioner and respondent.

Judgement: The final verdict, operative orders, and sentencing details.

Statute: Explicit references to laws, acts, sections, and constitutional articles cited.

PDF Processing Pipeline: A robust pre-processing module built to handle raw PDF judgments sourced directly from the Supreme Court of India. It includes advanced cleaning routines to strip header/footer noise, handle page breaks, and perform accurate sentence tokenization on noisy data.

Advanced Evaluation: Validated using a comprehensive suite of metrics that measure both lexical overlap (ROUGE, METEOR, BLEU) and semantic similarity (BERTScore), ensuring the summaries are not just copying words but retaining meaning.

📂 Repository Structure

Legal-Judgement-Summarizer-using-RL/
├── Model 1 (IN-EXT DATASET).ipynb         # Baseline model trained on 50 clean, text-based judgments
├── Model 2 (Indian kanoon).ipynb          # Advanced model trained on 400+ noisy PDF judgments (Supreme Court 2025)
├── Results and comparison of models.ipynb  # Side-by-side performance analysis comparing ROUGE/BLEU scores
├── valdiationinlegalbert50.ipynb          # Deep-dive validation script, embedding resizing logic, and BERTScore Radar Charts
├── STREAMLIT FILES/                       # Frontend UI components for the interactive web application (Work in Progress)
└── README.md                              # Comprehensive Project Documentation


🧠 Architecture & Methodology

The system employs an Actor-Critic style Reinforcement Learning approach where the "Policy Network" acts as the agent, making selection decisions based on the document state.

1. The Policy Network (The Agent)

The agent is responsible for reading the document and deciding which sentences are worth keeping.

Encoder: InLegalBERT processes sentences to produce dense, contextual embeddings that represent the semantic meaning of every sentence.

Context Layer: A Bi-directional LSTM (2 layers, 256 hidden dim) processes these embeddings to capture the global flow of the document, understanding how a sentence relates to the ones before and after it.

Attention Mechanism: A Multi-head attention layer allows the model to weigh sentences differently based on the requested aspect (e.g., focusing heavily on "Section" or "Act" keywords when asked for "Statute," while ignoring them for "Facts").

Action: The network outputs a probability distribution over all sentences, sampling them to form a candidate summary.

2. The Reward Function (The Critic)

Since we lack ground truth summaries during training, the model optimizes a composite Reference-Free Reward that mathematically defines "quality":

$$R_{total} = \alpha(Coh) + \beta(Cov) + \gamma(Div) - \delta(Red) + \epsilon(Info) + \zeta(Pos)$$

Component

Description

Improvement in Model 2

Coherence ($Coh$)

Measures cosine similarity between consecutive summary sentences to ensure a smooth narrative flow, preventing disjointed summaries.

Weight increased to 0.30 to prioritize readability.

Coverage ($Cov$)

Calculates how well the summary embeddings semantically cover the original document's embeddings, ensuring no key topics are lost.

Weight set to 0.25 for better information retention.

Redundancy ($Red$)

Penalizes the selection of semantically similar sentences to enforce conciseness and reduce repetitive legal jargon.

Penalty increased (0.20) to force tighter summaries.

Diversity ($Div$)

Encourages a wider spread of information by measuring the average pairwise distance between selected sentences.

Weight increased (0.15) to capture broader context.

Informativeness ($Info$)

Measures the "centrality" of selected sentences to ensure the most important, central ideas are picked.

Standard weight maintained.

Ordering ($Ord$)

New in Model 2: Penalties for selecting sentences out of chronological order, crucial for maintaining the timeline in "Facts".

Added feature for logical consistency.

📊 Models & Datasets

Model 1: IN-Ext Baseline

Dataset: 50 Text files from the IN-Ext Legal Dataset.

Objective: Served as a proof of concept to validate the aspect-based separation logic on clean data.

Limitations: constrained by fixed length inputs and a standard reward function that didn't account for sentence ordering.

Model 2: Scaled Indian Kanoon (Advanced)

Dataset: 400+ PDF files sourced directly from the Supreme Court of India (2025 data), representing real-world noise and variability.

Optimizations:

Memory Management: Implements torch.cuda.empty_cache() and dynamic sentence truncation (500-2000 limit) to fit massive legal documents on standard GPUs (Tesla T4).

Warmup Scheduler: A learning rate warmup for the first 3 epochs prevents the LSTM weights from diverging early in training.

Improved Rewards: Fine-tuned weights that heavily favor coherence and strictly penalize redundancy, producing summaries that read more naturally.

📈 Performance & Results

We evaluated both models using a hold-out test set. While Model 1 (trained on fewer, cleaner text files) achieves slightly higher raw metric scores due to the simplicity of its training domain, Model 2 demonstrates far superior generalization capabilities when dealing with real-world, noisy PDF inputs.

Metrics Comparison

Metric

Model 1 (50 Docs)

Model 2 (400 Docs)

Interpretation

BERTScore (F1)

0.832

0.828

Both models achieve high semantic similarity to human references, indicating they capture the core meaning well.

ROUGE-1

0.381

0.375

Competitive word-overlap performance comparable to supervised baselines in the legal domain.

Coherence

0.804

0.804

Identical semantic flow performance, proving the LSTM successfully models document structure.

Compression Ratio

~9.0%

~8.7%

Model 2 produces more concise summaries, filtering out more noise and boilerplate text.

Detailed visualization charts, including Radar plots and Aspect-wise performance breakdowns, are generated in valdiationinlegalbert50.ipynb.

🛠️ Installation & Usage

Prerequisites

Python 3.8+

GPU recommended (NVIDIA Tesla T4 or better for efficient training)

Install Dependencies

pip install torch transformers rouge-score bert-score nltk pandas matplotlib seaborn pdfplumber sentencepiece


Training

To train the advanced model on your own custom dataset of legal PDFs:

Place your PDF files in a designated folder.

Update the directory path in Model 2 (Indian kanoon).ipynb.

Run the notebook. The model will automatically preprocess the PDFs and save checkpoints as .pt files.

Evaluation

To evaluate a trained model or generate summaries for new documents:

from model_utils import load_trained_model, UnsupervisedRLAgent

# Load the checkpoint
agent, _ = load_trained_model('final_inlegalbert_model.pt')

# Generate summary
judgment_text = "..." # Your extracted legal text
summary = agent.generate_summaries(judgment_text)

print("Facts Summary:", summary['facts'])
print("Final Judgment:", summary['judgement'])


🤝 Acknowledgements

InLegalBERT: Thanks to Law-AI for providing the pre-trained model that forms the backbone of our encoder.

Indian Kanoon: For providing invaluable access to the legal judgments used for training and validation.
