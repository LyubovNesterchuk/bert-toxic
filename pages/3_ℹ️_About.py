import streamlit as st
import pandas as pd

from src.style import apply_styles

apply_styles()

# PAGE CONFIG
st.set_page_config(
    page_title="ToxicGuard — About",
    page_icon="ℹ️",
    layout="wide",
)


# HEADER

st.title("ℹ️ About ToxicGuard")

st.subheader(
    "BERT-based Toxic Comment Detection"
)


st.write(
    """
    **ToxicGuard** is an AI-powered text classification
    application designed to identify different types
    of toxic content in user comments.
    """
)


st.divider()


# PROJECT GOAL

st.markdown(
    "### 🎯 Project goal"
)

st.write(
    """
    The goal of the project is to develop a model capable
    of identifying and classifying different categories
    of toxicity in text comments using BERT-based
    natural language processing.
    """
)


# MODEL

st.markdown(
    "### 🤖 Model"
)

model_col1, model_col2 = st.columns(2)

with model_col1:

    st.metric(
        "Architecture",
        "BERT",
    )

    st.metric(
        "Task",
        "Multi-label classification",
    )

    st.metric(
        "Toxicity categories",
        "6",
    )


with model_col2:

    st.metric(
        "Maximum sequence length",
        "286",
    )

    st.metric(
        "Classification",
        "6 independent labels",
    )

    st.metric(
        "Decision",
        "Optimized thresholds",
    )


st.divider()



# TOXICITY CATEGORIES

st.markdown(
    "### 🏷️ Toxicity categories"
)


categories = {
    "toxic":
        "General toxic or abusive content",

    "severe_toxic":
        "Severely toxic content",

    "obscene":
        "Obscene or vulgar language",

    "threat":
        "Threatening content",

    "insult":
        "Insults or offensive statements",

    "identity_hate":
        "Hateful content targeting identity groups",
}


for label, description in categories.items():

    st.write(
        f"**{label.replace('_', ' ').title()}** — "
        f"{description}"
    )


st.divider()



# THRESHOLDS

st.markdown(
    "### ⚙️ Optimized decision thresholds"
)

st.write(
    """
    The application uses independently optimized
    decision thresholds for each toxicity category
    instead of applying the same 0.50 threshold to
    every class.
    """
)

thresholds = {
    "toxic": 0.90,
    "severe_toxic": 0.95,
    "obscene": 0.95,
    "threat": 0.80,
    "insult": 0.90,
    "identity_hate": 0.95
}


threshold_columns = st.columns(3)


for index, label in enumerate(thresholds):

    with threshold_columns[index % 3]:

        st.metric(
            label.replace("_", " ").title(),
            f"{thresholds[label]:.2f}",
        )


st.divider()


# TECHNOLOGY STACK

st.markdown(
    "### 🛠️ Technology stack"
)

technologies = [
    "Python",
    "PyTorch",
    "Hugging Face Transformers",
    "BERT",
    "scikit-learn",
    "Pandas",
    "Streamlit",
]

st.write(
    " • ".join(technologies)
)

st.divider()



# BASELINE

st.markdown(
    "### ⚖️ Baseline model"
)

st.write(
    """
    The BERT model was evaluated against a classical
    TF-IDF + Logistic Regression baseline.

    This comparison helps determine whether the
    transformer-based approach provides an advantage
    over traditional text classification methods.
    """
)

st.divider()


# LIMITATIONS

st.markdown(
    "### ⚠️ Limitations"
)

st.write(
    """
    Model predictions should be interpreted as automated
    classification rather than a definitive judgment
    of a comment.

    Toxicity detection can be affected by context,
    sarcasm, spelling variations, ambiguous language,
    and comments that contain multiple meanings.
    """
)

st.divider()


st.caption(
    "ToxicGuard • BERT Multi-label Toxicity Detection"
)

