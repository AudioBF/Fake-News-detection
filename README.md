# Fake News Detection with BERT

Binary classification of news articles as **real** or **fake** using fine-tuned `bert-base-uncased`.  
Served via a **Flask REST API** with a dark-themed web interface for interactive inference.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

---

## Results

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|----------|--------------|
| 1 | 0.0079 | 0.0000 | **100.00%** |
| 2 | 0.0006 | 0.0009 | 99.93% |
| 3 | 0.0005 | 0.0000 | **100.00%** |

**Best model saved at Epoch 3 · Val Accuracy: 100.00%**

### Training History

![Training and Validation Loss](training_history.png)

Training loss drops from `0.009` → `0.0005` across 3 epochs. Validation loss converges to near-zero, confirming stable fine-tuning without overfitting.

---

## On Data Leakage — Investigation & Decision

The Kaggle Fake/Real News dataset contains metadata columns beyond article text: `title`, `subject`, and `date`. During development I investigated potential **data leakage**:

- The `subject` column differs **systematically** between real (`politicsNews`, `worldnews`) and fake (`News`, `politics`, `left-news`) articles
- A model trained on `subject` alone would achieve near-perfect accuracy without reading article content

**Action taken:** All metadata columns (`title`, `subject`, `date`) were dropped before training. Only raw article `text` is used.

High accuracy persisted — consistent with known characteristics of this benchmark, where writing style differs significantly between Reuters wire format and unstructured fake content. This was a deliberate architectural decision, not an oversight.

---

## Architecture

```
True.csv + Fake.csv
        │
        ▼
  load_data()
  ├── Drop: title, subject, date  ← leakage prevention
  ├── Label: real=1, fake=0
  └── Shuffle (random_state=42)
        │
        ▼
  Train / Val Split (80% / 20%)
        │
        ▼
  BertTokenizer
  └── max_length=512 · padding · truncation
        │
        ▼
  NewsDataset (PyTorch) → DataLoader (batch=16)
        │
        ▼
  BertForSequenceClassification
  ├── Base: bert-base-uncased
  ├── Optimizer: AdamW (lr=2e-5)
  ├── Scheduler: Linear warmup
  ├── Gradient clipping (max_norm=1.0)
  └── Saves best_model.pt at peak val accuracy
        │
        ▼
  FakeNewsClassifier (inference.py)
        │
        ▼
  Flask REST API (app.py)
  └── POST /predict → {"prediction": "REAL"|"FAKE", "confidence": "99.81%"}
```

---

## Project Structure

```
fake-news-detection/
│
├── train.py                    # Full training pipeline
├── inference.py                # FakeNewsClassifier class
├── predict.py                  # CLI prediction script
├── app.py                      # Flask REST API
│
├── templates/
│   └── index.html              # Web interface
│
├── fake_news_classifier.ipynb  # Exploratory notebook
├── fake_news_detection.ipynb   # Training experiments
│
├── True.csv                    # Real news dataset (labeled 1)
├── Fake.csv                    # Fake news dataset (labeled 0)
├── training_history.png        # Loss curves (generated after training)
├── requirements.txt
└── .gitignore
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/AudioBF/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

Outputs:
- `best_model.pt` — trained model weights
- `training_history.png` — loss curves

> ⚠️ GPU strongly recommended. Trained on RTX 5060 Ti (~20 min). CPU fallback supported but slow.

### 3. Save model for inference

Run this once after training to convert `best_model.pt` to the HuggingFace format used by `inference.py`:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
model.save_pretrained('fake_news_model')
tokenizer.save_pretrained('fake_news_model')
```

### 4. Run the Flask app

```bash
python app.py
```

Open `http://localhost:5000` — paste any news article and classify it instantly.

### 5. CLI prediction

```bash
python predict.py --text "Your news article text here"
# Output: Prediction: REAL
```

### 6. REST API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WASHINGTON (Reuters) - The Federal Reserve held interest rates steady."}'
```

```json
{
  "prediction": "REAL",
  "confidence": "99.98%"
}
```

---

## Key Technical Decisions

**Leakage investigation before reporting results** — metadata columns audited and removed. Accuracy without leakage is the honest benchmark.

**BERT over TF-IDF + classical ML** — transformer attention captures long-range semantic dependencies that bag-of-words approaches miss. Critical for detecting subtle stylistic differences in news writing.

**AdamW + linear warmup at lr=2e-5** — standard BERT fine-tuning configuration. Prevents catastrophic forgetting of pre-trained representations while adapting to the classification task.

**Gradient clipping (max_norm=1.0)** — stabilizes training with long sequences (max_length=512).

**Flask over Streamlit** — REST architecture allows the classifier to be consumed as a service, not just a standalone demo. More representative of production deployment patterns.

**Separate inference module** — `FakeNewsClassifier` class in `inference.py` decouples model loading from the API layer, making the system extensible and testable independently.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | `bert-base-uncased` (Hugging Face Transformers) |
| Training | PyTorch 2.12 · AdamW · Linear LR Scheduler |
| API | Flask 3.1 |
| Data | Pandas · NumPy · Scikit-learn |
| Visualization | Matplotlib · Seaborn |
| Hardware | NVIDIA RTX 5060 Ti · CUDA 12.8 |

---

## Author

**Audio Fagundes** — AI Software Engineer · Dublin, Ireland

[![LinkedIn](https://img.shields.io/badge/LinkedIn-audiofagundes-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/audiofagundes/)
[![GitHub](https://img.shields.io/badge/GitHub-AudioBF-181717?style=flat&logo=github&logoColor=white)](https://github.com/AudioBF)
[![Email](https://img.shields.io/badge/Email-audiobf@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:audiobf@gmail.com)

---

## License

MIT License