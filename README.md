# 🛡️ ToxicGuard

## AI-powered Toxic Comment Detection

**ToxicGuard** is a machine learning application for detecting and classifying toxic comments using a fine-tuned **BERT** model.

The application performs multi-label classification and detects six types of toxicity:

* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`

The project includes data analysis, a TF-IDF baseline, BERT fine-tuning, class-imbalance handling, threshold optimization, model evaluation, and a Streamlit web interface.

---

## 🎯 Project Goal

The goal of the project is to develop an NLP system capable of detecting different types of toxic language in online comments.

The main challenges addressed are:

* **Context understanding** — using BERT to capture contextual relationships in text.
* **Class imbalance** — using weighted loss and individual classification thresholds for rare toxicity classes.

---

## 📊 Dataset

The project uses the **Jigsaw Toxic Comment Classification Challenge** dataset.

The dataset contains approximately 160,000 Wikipedia comments labeled with six toxicity categories.

| Label           | Description                     |
| --------------- | ------------------------------- |
| `toxic`         | Toxic or abusive language       |
| `severe_toxic`  | Severely toxic language         |
| `obscene`       | Obscene language                |
| `threat`        | Threatening language            |
| `insult`        | Insulting language              |
| `identity_hate` | Identity-based hateful language |

After data preparation and duplicate handling, the dataset contained **159,403 comments**.

The data was divided into:

| Split      | Samples |
| ---------- | ------: |
| Train      | 127,523 |
| Validation |  15,940 |
| Test       |  15,940 |

---

## 🤖 Models

### TF-IDF Baseline

A traditional NLP baseline was created using:

* TF-IDF
* Logistic Regression
* Multi-label classification

**Macro F1: 0.6069**

### BERT

The main model is based on:

**`bert-base-uncased`**

The model was fine-tuned for six-label multi-label classification.

To address class imbalance, the training process used:

* `BCEWithLogitsLoss`
* positive class weights (`pos_weight`)
* label-specific classification thresholds

Final validation thresholds:

| Label           | Threshold |
| --------------- | --------: |
| `toxic`         |      0.90 |
| `severe_toxic`  |      0.95 |
| `obscene`       |      0.95 |
| `threat`        |      0.80 |
| `insult`        |      0.90 |
| `identity_hate` |      0.95 |

---

## 🏆 Final Results

The final BERT model was evaluated on the independent test set.

| Metric        |       BERT |
| ------------- | ---------: |
| Macro F1      | **0.6809** |
| Micro F1      | **0.7887** |
| Weighted F1   | **0.7937** |
| Macro ROC-AUC | **0.9913** |
| Macro PR-AUC  | **0.7205** |

### BERT vs TF-IDF

| Model                        |   Macro F1 |
| ---------------------------- | ---------: |
| TF-IDF + Logistic Regression |     0.6069 |
| **BERT**                     | **0.6809** |

BERT improved Macro F1 by **0.0740**, or approximately **7.4 percentage points**.

---

## 🔬 Error Analysis

The project also includes error analysis of:

* **False Positives** — non-toxic comments incorrectly classified as toxic.
* **False Negatives** — toxic comments incorrectly classified as non-toxic.

This analysis helps identify difficult linguistic cases and possible areas for future model improvement.

---

## 🖥️ Streamlit Application

ToxicGuard provides a Streamlit interface where users can enter a comment and receive toxicity predictions.

The application displays the detected toxicity categories and their prediction probabilities.

Run locally with:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
bert-toxic/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── README.md
│
├── src/
│   └── prediction.py
│
└── models/
    └── bert-toxic/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── thresholds.json
```

---

## 🐳 Docker

The project includes a `Dockerfile` for containerized deployment.

### Build the Docker image

```bash
docker build -t toxicguard .
```

### Run the container

```bash
docker run -p 8501:8501 toxicguard
```

Open:

```text
http://localhost:8501
```

---

## 🐳 Docker Compose

Docker Compose simplifies building and running the application.

Start the application with one command:

```bash
docker compose up --build
```

The application will be available at:

```text
http://localhost:8501
```

To run it in the background:

```bash
docker compose up -d
```

To stop the application:

```bash
docker compose down
```

The Compose configuration mounts the local model directory:

```yaml
volumes:
  - ./models:/app/models
```

This allows the container to access the trained BERT model stored in the project.

---

## 🧰 Technologies

* Python 3.11
* PyTorch
* Hugging Face Transformers
* BERT
* scikit-learn
* Pandas
* NumPy
* Streamlit
* Docker
* Docker Compose

---

👩‍💻 Author

Lyubov Nesterchukl

Machine Learning / NLP project developed as part of a practical machine learning and application deployment workflow.

📄 License

This project is intended for educational and demonstration purposes.
