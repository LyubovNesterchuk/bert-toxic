import streamlit as st
import pandas as pd
import plotly.express as px

from src.style import apply_styles

apply_styles()

# PAGE CONFIG
st.set_page_config(
    page_title="ToxicGuard — Model Performance",
    page_icon="📊",
    layout="wide",
)



# VALIDATION RESULTS BERT vs TF-IDF AFTER THRESHOLD TUNING
validation_results = {
    "toxic": {
        "precision": 0.833755,
        "recall": 0.863220,
        "f1": 0.848232,
        "support": 1528,
        "tfidf_f1": 0.776479,
    },
    "severe_toxic": {
        "precision": 0.433476,
        "recall": 0.635220,
        "f1": 0.515306,
        "support": 159,
        "tfidf_f1": 0.504902,
    },
    "obscene": {
        "precision": 0.852732,
        "recall": 0.851720,
        "f1": 0.852226,
        "support": 843,
        "tfidf_f1": 0.809242,
    },
    "threat": {
        "precision": 0.450980,
        "recall": 0.489362,
        "f1": 0.469388,
        "support": 47,
        "tfidf_f1": 0.426966,
    },
    "insult": {
        "precision": 0.745495,
        "recall": 0.841169,
        "f1": 0.790448,
        "support": 787,
        "tfidf_f1": 0.712963,
    },
    "identity_hate": {
        "precision": 0.618750,
        "recall": 0.707143,
        "f1": 0.660000,
        "support": 140,
        "tfidf_f1": 0.410959,
    },
}



# FINAL BERT TEST RESULTS
test_metrics = {
    "Macro F1": 0.6875,
    "Micro F1": 0.7937,
    "Weighted F1": 0.7968,
    "ROC-AUC Macro": 0.9909,
    "PR-AUC Macro": 0.7217,
}


test_results = {
    "toxic": {
        "precision": 0.83,
        "recall": 0.85,
        "f1": 0.84,
        "support": 1528,
    },
    "severe_toxic": {
        "precision": 0.45,
        "recall": 0.62,
        "f1": 0.52,
        "support": 160,
    },
    "obscene": {
        "precision": 0.84,
        "recall": 0.84,
        "f1": 0.84,
        "support": 845,
    },
    "threat": {
        "precision": 0.52,
        "recall": 0.67,
        "f1": 0.58,
        "support": 48,
    },
    "insult": {
        "precision": 0.73,
        "recall": 0.82,
        "f1": 0.77,
        "support": 788,
    },
    "identity_hate": {
        "precision": 0.54,
        "recall": 0.59,
        "f1": 0.56,
        "support": 140,
    },
}



# HEADER

st.title("📊 Toxic Comment Classification — Model Performance")

st.caption(
    "Comparison of the BERT model with the TF-IDF + Logistic Regression "
    "baseline, followed by final BERT evaluation on the test set."
)

st.divider()



# FINAL MODEL

st.markdown("## 🤖 Final model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "BERT")

with col2:
    st.metric("Toxicity categories", "6")

with col3:
    st.metric("Threshold tuning", "Per class")


st.caption(
    "Thresholds were optimized separately for each toxicity category "
    "on the validation set."
)

st.divider()



# VALIDATION COMPARISON

st.markdown("## 🎯 BERT vs TF-IDF + Logistic Regression")

st.caption(
    "Validation results after threshold tuning. "
    "F1-score is used as the primary comparison metric because the dataset "
    "contains strong class imbalance."
)


comparison_data = []

for label, values in validation_results.items():

    change = values["f1"] - values["tfidf_f1"]

    comparison_data.append(
        {
            "Category": label,
            "BERT F1": values["f1"],
            "TF-IDF F1": values["tfidf_f1"],
            "F1 improvement": change,
            "Improvement %": (
                change / values["tfidf_f1"] * 100
            ),
        }
    )


comparison_df = pd.DataFrame(comparison_data)

st.dataframe(
    comparison_df.style.format(
        {
            "BERT F1": "{:.4f}",
            "TF-IDF F1": "{:.4f}",
            "F1 improvement": "{:+.4f}",
            "Improvement %": "{:+.2f}%",
        }
    ),
    use_container_width=True,
    hide_index=True,
)



# F1 CHART

st.markdown("### 📊 BERT vs TF-IDF — F1-score by category")

st.caption(
    "Comparison of F1-score for each toxicity category "
    "on the validation set after threshold tuning."
)

chart_df = comparison_df[
    ["Category", "BERT F1", "TF-IDF F1"]
].copy()


chart_long = chart_df.melt(
    id_vars="Category",
    value_vars=["BERT F1", "TF-IDF F1"],
    var_name="Model",
    value_name="F1",
)


fig = px.bar(
    chart_long,
    x="F1",
    y="Category",
    color="Model",
    barmode="group",
    orientation="h",
    text="F1",
)


# Show F1 value on each bar
fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
)


fig.update_layout(
    xaxis_title="F1 score",
    yaxis_title="",
    xaxis=dict(
        range=[0, 1],
        tickformat=".1f",
    ),
    legend_title="Model",
    height=500,
    margin=dict(
        l=20,
        r=80,
        t=30,
        b=20,
    ),
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# KEY FINDINGS

st.markdown("### 💡 Key findings")

best_category = comparison_df.loc[
    comparison_df["F1 improvement"].idxmax()
]

st.success(
    f"BERT outperforms TF-IDF on all six toxicity categories. "
    f"The largest improvement is for "
    f"**{best_category['Category']}**: "
    f"+{best_category['F1 improvement']:.4f} F1 "
    f"(+{best_category['Improvement %']:.2f}%)."
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Categories won by BERT",
        "6 / 6",
    )

with col2:
    st.metric(
        "Largest F1 improvement",
        f"+{best_category['F1 improvement']:.4f}",
    )

with col3:
    st.metric(
        "Largest improvement",
        best_category["Category"],
    )


st.divider()



# VALIDATION CLASSIFICATION DETAILS


st.markdown("## 🔎 BERT validation classification details")

validation_report_df = pd.DataFrame(
    [
        {
            "Category": label,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1"],
            "Support": values["support"],
        }
        for label, values in validation_results.items()
    ]
)


st.dataframe(
    validation_report_df.style.format(
        {
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "Support": "{:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


st.divider()



# FINAL BERT TEST RESULTS

st.markdown("## 🧪 Final BERT test results")

st.caption(
    "Final evaluation of the selected BERT model on the test set."
)


test_columns = st.columns(5)

for index, (metric, value) in enumerate(test_metrics.items()):

    with test_columns[index]:

        st.metric(
            metric,
            f"{value:.4f}",
        )



# TEST CLASSIFICATION REPORT

st.markdown("### 🎯 BERT test classification report")

test_report_df = pd.DataFrame(
    [
        {
            "Category": label,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1"],
            "Support": values["support"],
        }
        for label, values in test_results.items()
    ]
)


st.dataframe(
    test_report_df.style.format(
        {
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1": "{:.2f}",
            "Support": "{:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# TEST F1 CHART

st.markdown("### 📊 Final BERT F1-score by toxicity category")

test_f1_chart = test_report_df.set_index(
    "Category"
)[["F1"]]


st.bar_chart(
    test_f1_chart,
    y_label="F1 score",
)



# TEST SUMMARY

st.markdown("### 💡 Final test evaluation")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Macro F1",
        f"{test_metrics['Macro F1']:.4f}",
    )

with col2:
    st.metric(
        "Micro F1",
        f"{test_metrics['Micro F1']:.4f}",
    )

with col3:
    st.metric(
        "Weighted F1",
        f"{test_metrics['Weighted F1']:.4f}",
    )


st.info(
    "The final BERT model demonstrates strong overall classification "
    "performance on the test set. The Macro F1 of 0.6875 reflects "
    "performance across all toxicity categories equally, while the "
    "Weighted F1 of 0.7968 accounts for the different class frequencies."
)

st.divider()


st.caption(
    "ToxicGuard • BERT Multi-label Toxicity Detection"
)