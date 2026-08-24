import streamlit as st

from src.style import apply_styles

apply_styles()

st.set_page_config(
    page_title="ToxicGuard",
    page_icon="🛡️",
    layout="wide",
)


st.title("🛡️ ToxicGuard")

st.subheader(
    "AI-powered Toxic Comment Detection"
)

st.write(
    """
    Welcome to **ToxicGuard** — a BERT-based application
    for detecting and classifying toxic comments.
    """
)

st.divider()


st.markdown("### 🚀 What can you do?")


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("### 🔍 Analyze")

    st.write(
        """
        Enter a comment and let the BERT model
        identify different types of toxicity.
        """
    )


with col2:

    st.markdown("### 📊 Model Performance")

    st.write(
        """
        Compare BERT with the
        TF-IDF + Logistic Regression baseline.
        """
    )


with col3:

    st.markdown("### ℹ️ About")

    st.write(
        """
        Learn about the model, dataset,
        thresholds and technologies.
        """
    )


st.divider()


st.info(
    "Use the navigation menu on the left to explore ToxicGuard."
)


st.caption(
    "ToxicGuard • BERT Multi-label Toxicity Detection"
)



# import streamlit as st
# from src.prediction import predict, LABELS


# st.set_page_config(
#     page_title="Toxic Comment Detector",
#     page_icon="⚠️",
#     layout="centered"
# )


# st.title("⚠️ Toxic Comment Detector")

# st.write(
#     "BERT-модель для визначення токсичності коментарів."
# )


# text = st.text_area(
#     "Введіть коментар:",
#     height=150,
#     placeholder="Наприклад: You are a terrible person..."
# )


# if st.button("🔍 Аналізувати", type="primary"):

#     if not text.strip():
#         st.warning("Будь ласка, введіть текст.")
#     else:

#         with st.spinner("Аналізую..."):

#             results = predict(text)

#         st.subheader("Результат")

#         detected_labels = [
#             label
#             for label in LABELS
#             if results[label]["detected"]
#         ]

#         if detected_labels:
#             st.error(
#                 "Виявлено токсичність: "
#                 + ", ".join(detected_labels)
#             )
#         else:
#             st.success("Токсичність не виявлена.")

#         st.subheader("Детальні результати")

#         for label in LABELS:

#             result = results[label]

#             probability = result["probability"]
#             threshold = result["threshold"]

#             st.write(
#                 f"**{label}** — "
#                 f"{probability:.2%} "
#                 f"(threshold: {threshold:.2f})"
#             )

#             st.progress(float(probability))



# в терміналі
# streamlit run app.py

# тестуємо мінімум 4 коментарі:

# 1. Явно токсичний
# You are an idiot and I hate you.
# 2. З погрозою
# I will find you and hurt you.
# 3. З образою
# You are stupid and completely useless.
# 4. Нейтральний
# I really enjoyed reading this article.

