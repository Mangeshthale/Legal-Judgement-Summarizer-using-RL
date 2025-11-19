# Legal Judgment Summarizer using Unsupervised Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A domain-specific, unsupervised abstractive/extractive summarization system designed specifically for **Indian Legal Judgments**. This project utilizes **Deep Reinforcement Learning (Policy Gradient)** to generate highly structured, multi-aspect summaries (Facts, Analysis, Arguments, Judgment, Statute), overcoming the critical bottleneck of requiring ground-truth reference summaries during the training phase.

---

## 🚀 Project Overview

Legal documents, particularly Indian court judgments, are notoriously long, complex, and unstructured. They often contain archaic terminology, intricate sentence structures, and non-linear narratives that make manual summarization a time-consuming and expensive task.

Traditional supervised summarization models fail in this domain because they require massive datasets of paired (document, summary) examples, which are scarce and costly to create in the legal field.

### The Solution

This project solves that **data scarcity bottleneck** using an **Unsupervised RL Agent**. Instead of mimicking human summaries, the agent learns to identify and select the most salient sentences by continuously optimizing a **Reference-Free Reward Function**. This function is composed of intrinsic quality metrics—such as coherence, coverage, and diversity—allowing the model to "teach itself" what a good summary looks like without ever seeing one.

---

## ✨ Key Features

### 🎯 Domain-Specific Encoder
Leveraging **law-ai/InLegalBERT** (frozen) to generate 768-dimensional embeddings. Unlike generic BERT models, this encoder is pre-trained on Indian legal texts, allowing it to capture nuances like "petitioner," "respondent," and specific acts with greater semantic accuracy.

### 📑 Multi-Aspect Summarization
The model goes beyond generic summaries by segmenting the judgment into **5 distinct, professionally relevant aspects**:

| Aspect | Description |
|--------|-------------|
| **Facts** | The procedural history and background story leading to the dispute |
| **Analysis** | The court's logical reasoning and application of legal principles |
| **Argument** | The specific contentions and counter-claims presented by the petitioner and respondent |
| **Judgement** | The final verdict, operative orders, and sentencing details |
| **Statute** | Explicit references to laws, acts, sections, and constitutional articles cited |

### 📄 PDF Processing Pipeline
A robust pre-processing module built to handle raw PDF judgments sourced directly from the Supreme Court of India. It includes advanced cleaning routines to strip header/footer noise, handle page breaks, and perform accurate sentence tokenization on noisy data.

### 📊 Advanced Evaluation
Validated using a comprehensive suite of metrics that measure both lexical overlap (ROUGE, METEOR, BLEU) and semantic similarity (BERTScore), ensuring the summaries are not just copying words but retaining meaning.

---

## 📂 Repository Structure

```
Legal-Judgement-Summarizer-using-RL/
├── Model 1 (IN-EXT DATASET).ipynb          # Baseline model trained on 50 clean, text-based judgments
├── Model 2 (Indian kanoon).ipynb           # Advanced model trained on 400+ noisy PDF judgments
├── Results and comparison of models.ipynb  # Side-by-side performance analysis
├── valdiationinlegalbert50.ipynb          # Deep-dive validation script with BERTScore Radar Charts
├── STREAMLIT FILES/                        # Frontend UI components (Work in Progress)
└── README.md                               # This file
```

---

## 🧠 Architecture & Methodology

The system employs an **Actor-Critic style Reinforcement Learning** approach where the "Policy Network" acts as the agent, making selection decisions based on the document state.

### 1. The Policy Network (The Agent)

The agent is responsible for reading the document and deciding which sentences are worth keeping.

- **Encoder**: InLegalBERT processes sentences to produce dense, contextual embeddings that represent the semantic meaning of every sentence
- **Context Layer**: A Bi-directional LSTM (2 layers, 256 hidden dim) processes these embeddings to capture the global flow of the document
- **Attention Mechanism**: A Multi-head attention layer allows the model to weigh sentences differently based on the requested aspect
- **Action**: The network outputs a probability distribution over all sentences, sampling them to form a candidate summary

### 2. The Reward Function (The Critic)

Since we lack ground truth summaries during training, the model optimizes a composite **Reference-Free Reward**:

$$R_{total} = \alpha(Coh) + \beta(Cov) + \gamma(Div) - \delta(Red) + \epsilon(Info) + \zeta(Pos)$$

| Component | Description | Improvement in Model 2 |
|-----------|-------------|------------------------|
| **Coherence (Coh)** | Measures cosine similarity between consecutive summary sentences to ensure smooth narrative flow | Weight increased to 0.30 to prioritize readability |
| **Coverage (Cov)** | Calculates how well the summary embeddings semantically cover the original document's embeddings | Weight set to 0.25 for better information retention |
| **Redundancy (Red)** | Penalizes the selection of semantically similar sentences to enforce conciseness | Penalty increased (0.20) to force tighter summaries |
| **Diversity (Div)** | Encourages a wider spread of information by measuring average pairwise distance between sentences | Weight increased (0.15) to capture broader context |
| **Informativeness (Info)** | Measures the "centrality" of selected sentences to ensure the most important ideas are picked | Standard weight maintained |
| **Ordering (Ord)** | **New in Model 2**: Penalties for selecting sentences out of chronological order | Added feature for logical consistency |

---

## 📊 Models & Datasets

### Model 1: IN-Ext Baseline

- **Dataset**: 50 Text files from the IN-Ext Legal Dataset
- **Objective**: Served as a proof of concept to validate the aspect-based separation logic on clean data
- **Limitations**: Constrained by fixed length inputs and a standard reward function that didn't account for sentence ordering

### Model 2: Scaled Indian Kanoon (Advanced)

- **Dataset**: 400+ PDF files sourced directly from the Supreme Court of India (2025 data)
- **Optimizations**:
  - **Memory Management**: Implements `torch.cuda.empty_cache()` and dynamic sentence truncation (500-2000 limit)
  - **Warmup Scheduler**: Learning rate warmup for the first 3 epochs prevents LSTM weights from diverging
  - **Improved Rewards**: Fine-tuned weights that heavily favor coherence and strictly penalize redundancy

---

## 📈 Performance & Results

We evaluated both models using a hold-out test set. While Model 1 achieves slightly higher raw metric scores due to cleaner training data, Model 2 demonstrates **far superior generalization capabilities** when dealing with real-world, noisy PDF inputs.

### Metrics Comparison

| Metric | Model 1 (50 Docs) | Model 2 (400 Docs) | Interpretation |
|--------|-------------------|-------------------|----------------|
| **BERTScore (F1)** | 0.832 | 0.828 | Both models achieve high semantic similarity to human references |
| **ROUGE-1** | 0.381 | 0.375 | Competitive word-overlap performance comparable to supervised baselines |
| **Coherence** | 0.804 | 0.804 | Identical semantic flow performance |
| **Compression Ratio** | ~9.0% | ~8.7% | Model 2 produces more concise summaries, filtering out more noise |

> Detailed visualization charts, including Radar plots and Aspect-wise performance breakdowns, are generated in `valdiationinlegalbert50.ipynb`.

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.8+
- GPU recommended (NVIDIA Tesla T4 or better for efficient training)

### Install Dependencies

```bash
pip install torch transformers rouge-score bert-score nltk pandas matplotlib seaborn pdfplumber sentencepiece
```

### Training

To train the advanced model on your own custom dataset of legal PDFs:

1. Place your PDF files in a designated folder
2. Update the directory path in `Model 2 (Indian kanoon).ipynb`
3. Run the notebook

The model will automatically preprocess the PDFs and save checkpoints as `.pt` files.

### Evaluation

To evaluate a trained model or generate summaries for new documents:

```python
from model_utils import load_trained_model, UnsupervisedRLAgent

# Load the checkpoint
agent, _ = load_trained_model('final_inlegalbert_model.pt')

# Generate summary
judgment_text = "..."  # Your extracted legal text
summary = agent.generate_summaries(judgment_text)

print("Facts Summary:", summary['facts'])
print("Final Judgment:", summary['judgement'])
```

---

## 🎯 Use Cases

- **Legal Research**: Quickly extract key information from lengthy judgments
- **Case Law Analysis**: Identify relevant precedents and statutory references
- **Legal Education**: Help law students understand complex judgments
- **Document Management**: Organize and categorize large volumes of legal documents
- **Client Communication**: Generate accessible summaries for non-legal audiences

---

## 🔮 Future Work

- [ ] Expand to district and high court judgments
- [ ] Add support for multiple Indian languages
- [ ] Implement citation network analysis
- [ ] Deploy as a web service with the Streamlit interface
- [ ] Fine-tune with human feedback (RLHF)
- [ ] Add support for comparative legal analysis across judgments

---

## 🤝 Acknowledgements

- **InLegalBERT**: Thanks to [Law-AI](https://huggingface.co/law-ai/InLegalBERT) for providing the pre-trained model that forms the backbone of our encoder
- **Indian Kanoon**: For providing invaluable access to the legal judgments used for training and validation
- **Supreme Court of India**: For making judgments publicly accessible

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue or reach out to the project maintainers.

---

## ⭐ Citation

If you use this work in your research, please cite:

```bibtex
@software{legal_judgment_summarizer,
  title={Legal Judgment Summarizer using Unsupervised Reinforcement Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Legal-Judgement-Summarizer-using-RL}
}
```

---

<p align="center"></p>
