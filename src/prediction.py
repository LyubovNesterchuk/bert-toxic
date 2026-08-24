import json
import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_PATH = "models/bert-toxic"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@st.cache_resource(show_spinner="Loading BERT model...")
def load_model():

    # Load tokenizer ONLY from local files
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # Load model ONLY from local files
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    # Load thresholds
    with open(
        f"{MODEL_PATH}/thresholds.json",
        "r",
        encoding="utf-8",
    ) as f:
        thresholds = json.load(f)

    return tokenizer, model, thresholds


tokenizer, model, thresholds = load_model()


def predict(text, max_length=256):

    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoding = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    with torch.no_grad():

        outputs = model(**encoding)

    probabilities = (
        torch.sigmoid(outputs.logits)[0]
        .cpu()
        .numpy()
    )

    predictions = {}

    for label, probability in zip(
        LABELS,
        probabilities,
    ):

        threshold = float(
            thresholds[label]
        )

        predictions[label] = {
            "probability": float(probability),
            "threshold": threshold,
            "detected": bool(
                probability >= threshold
            ),
        }

    return predictions


# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


# MODEL_PATH = "models/bert-toxic"

# LABELS = [
#     "toxic",
#     "severe_toxic",
#     "obscene",
#     "threat",
#     "insult",
#     "identity_hate",
# ]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)



# # Load BERT model
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_PATH,
#     local_files_only=True
# )

# model.to(device)
# model.eval()



# # Load thresholds
# with open(
#     f"{MODEL_PATH}/thresholds.json",
#     "r",
#     encoding="utf-8"
# ) as f:
#     thresholds = json.load(f)


# def predict(text, max_length=384):
#     """
#     Predict toxicity labels for a single comment.
#     """

#     encoding = tokenizer(
#         text,
#         truncation=True,
#         padding=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )

#     encoding = {
#         key: value.to(device)
#         for key, value in encoding.items()
#     }

#     with torch.no_grad():
#         outputs = model(**encoding)

#     probabilities = torch.sigmoid(outputs.logits)[0].cpu().numpy()

#     predictions = {}

#     for label, probability in zip(LABELS, probabilities):

#         threshold = float(thresholds[label])

#         predictions[label] = {
#             "probability": float(probability),
#             "threshold": threshold,
#             "detected": bool(probability >= threshold)
#         }

#     return predictions
