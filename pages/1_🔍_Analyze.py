import streamlit as st
import random
from src.prediction import predict, LABELS

from src.style import apply_styles

apply_styles()

# PAGE CONFIG

st.set_page_config(
    page_title="ToxicGuard — Analyze",
    page_icon="🔍",
    layout="wide",
)



# SESSION STATE

if "history" not in st.session_state:
    st.session_state.history = []

if "text" not in st.session_state:
    st.session_state.text = ""


# HEADER

st.title("🔍 Analyze a Comment")

st.subheader(
    "AI-powered Toxic Comment Detection"
)

st.caption(
    "BERT-based multi-label classification "
    "with optimized decision thresholds"
)

st.divider()


# EXAMPLES

st.markdown("### 💬 Try an example")


EXAMPLES = {
    "🟢 Neutral": [
        "Thank you for your help. I really appreciate your answer.",
        "This article provides some useful information.",
        "I agree with your point about this topic.",
        "Could you please explain this in more detail?",
        "I enjoyed reading your comment.",
        "That is an interesting perspective.",
        "I think we should discuss this calmly.",
        "Thanks for sharing your opinion.",
        "This is a helpful suggestion.",
        "I understand what you mean.",
    ],

    "🟠 Toxic": [
        "You are absolutely disgusting.",
        "Nobody wants you here.",
        "You are such a horrible person.",
        "Your comments are completely toxic.",
        "I can't stand people like you.",
        "You make every discussion worse.",
        "What a terrible attitude you have.",
        "People like you ruin everything.",
        "You are making this conversation unbearable.",
        "You should be ashamed of yourself.",
    ],

    "🔴 Severe Toxic": [
        "You are a worthless piece of garbage.",
        "Everyone would be better off without you.",
        "You are absolutely pathetic and disgusting.",
        "Nobody could possibly respect someone like you.",
        "You are the worst kind of person.",
        "Your behavior is completely unacceptable.",
        "You are nothing but a miserable human being.",
        "I have never seen someone so hateful.",
        "You are completely vile and pathetic.",
        "People like you are beyond help.",
    ],

    "🟣 Obscene": [
        "What the hell are you talking about?",
        "This is complete bullshit.",
        "Stop talking such crap.",
        "That idea is fucking ridiculous.",
        "What a damn stupid argument.",
        "This whole thing is absolute crap.",
        "I don't give a damn about your opinion.",
        "That's a load of bullshit.",
        "This is getting really damn annoying.",
        "What the fuck is wrong with this discussion?",
    ],

    "🟡 Threat": [
        "You better watch your back.",
        "You will regret saying that.",
        "Someone should teach you a lesson.",
        "Keep talking and you will be sorry.",
        "You should be careful about who you insult.",
        "I know where you live.",
        "You won't get away with this.",
        "You better stay away from me.",
        "There will be consequences for what you said.",
        "Don't make me come after you.",
    ],

    "🔵 Insult": [
        "You are stupid and completely useless.",
        "What an idiot you are.",
        "You clearly have no idea what you're talking about.",
        "You are incredibly ignorant.",
        "That was a really dumb thing to say.",
        "You are such a fool.",
        "Your argument is ridiculous.",
        "You have absolutely no common sense.",
        "What a pathetic excuse for an argument.",
        "You are unbelievably clueless.",
    ],

    "🟤 Identity Hate": [
        "People from your group don't belong here.",
        "I can't stand people of your kind.",
        "Your group is ruining this country.",
        "People like you should not be allowed here.",
        "Your religion has no place in this community.",
        "I don't trust people from your background.",
        "Your kind of people are all the same.",
        "This community would be better without your group.",
        "People from your group are not welcome here.",
        "I wish people like you would leave this place.",
    ],
}


# 7 columns — one button for each category
columns = st.columns(4)

for i, (category, examples) in enumerate(EXAMPLES.items()):

    with columns[i % 4]:

        if st.button(
            category,
            use_container_width=True,
            key=f"example_{i}",
        ):

            # Random example from selected category
            st.session_state.text = random.choice(examples)

            st.rerun()


# TEXT INPUT

st.markdown("### 📝 Enter your comment")

st.markdown("""
<style>
textarea {
    font-size: 22px !important;
}
</style>
""", unsafe_allow_html=True)

text = st.text_area(
    "Comment",
    value=st.session_state.text,
    placeholder="Type or paste a comment here...",
    height=180,
    max_chars=5000,
)



# TEXT STATISTICS

characters = len(text)

words = (
    len(text.split())
    if text.strip()
    else 0
)


stat_col1, stat_col2, stat_col3 = st.columns(3)


with stat_col1:

    st.metric(
        "Characters",
        f"{characters:,}",
    )


with stat_col2:

    st.metric(
        "Words",
        f"{words:,}",
    )


with stat_col3:

    if characters == 0:
        status = "Empty"

    elif characters > 4000:
        status = "Long"

    else:
        status = "Ready"


    st.metric(
        "Status",
        status,
    )



# BUTTONS

button_col1, button_col2 = st.columns(2)


with button_col1:

    analyze = st.button(
        "🔍 Analyze comment",
        type="primary",
        use_container_width=True,
    )


with button_col2:

    clear = st.button(
        "🗑️ Clear",
        use_container_width=True,
    )


if clear:

    st.session_state.text = ""

    st.rerun()



# PREDICTION

if analyze:

    if not text.strip():

        st.warning(
            "Please enter a comment before "
            "starting the analysis."
        )

    else:

        with st.spinner(
            "Analyzing comment with BERT..."
        ):

            results = predict(text)


      
        # DETECTED LABELS
   
        detected_labels = [
            label
            for label in LABELS
            if results[label]["detected"]
        ]


       
        # OVERALL RESULT
        
        st.divider()

        st.markdown(
            "### 🎯 Overall assessment"
        )


        if detected_labels:

            st.error(
                "🔴 Toxic comment detected\n\n"
                "Detected categories: "
                + ", ".join(detected_labels)
            )

            overall_status = "Toxic"

        else:

            st.success(
                "🟢 Non-toxic comment\n\n"
                "No toxicity categories exceeded "
                "their optimized thresholds."
            )

            overall_status = "Non-toxic"


       
        # HISTORY
        
        history_item = {
            "text": text,
            "status": overall_status,
            "detected": detected_labels,
        }


        st.session_state.history.insert(
            0,
            history_item,
        )


        st.session_state.history = (
            st.session_state.history[:10]
        )


        
        # DETAILED ANALYSIS
        
        st.markdown(
            "### 📊 Toxicity analysis"
        )

        st.caption(
            "Probability and optimized threshold "
            "for each toxicity category."
        )


        columns = st.columns(3)


        for index, label in enumerate(LABELS):

            result = results[label]

            probability = result["probability"]
            threshold = result["threshold"]
            detected = result["detected"]


            with columns[index % 3]:

                st.markdown(
                    f"#### "
                    f"{label.replace('_', ' ').title()}"
                )


                st.metric(
                    "Probability",
                    f"{probability:.1%}",
                )


                st.progress(
                    min(probability, 1.0)
                )


                st.caption(
                    f"Threshold: {threshold:.2f}"
                )


                if detected:

                    st.error(
                        "🔴 Detected"
                    )

                else:

                    st.success(
                        "🟢 Not detected"
                    )


      
        # EXPLANATION
       
        if detected_labels:

            st.divider()

            st.markdown(
                "### 💡 Why was this comment flagged?"
            )


            for label in detected_labels:

                probability = (
                    results[label]["probability"]
                )

                threshold = (
                    results[label]["threshold"]
                )


                st.write(
                    f"**{label.replace('_', ' ').title()}** "
                    f"— probability **{probability:.1%}** "
                    f"exceeded the optimized threshold "
                    f"**{threshold:.2f}**."
                )



# HISTORY

if st.session_state.history:

    st.divider()

    st.markdown(
        "### 🕘 Recent analyses"
    )


    for item in st.session_state.history:

        preview = item["text"]


        if len(preview) > 80:

            preview = (
                preview[:80]
                + "..."
            )


        if item["status"] == "Toxic":

            icon = "🔴"

        else:

            icon = "🟢"


        with st.expander(
            f"{icon} {preview}"
        ):

            st.write(
                item["text"]
            )


            if item["detected"]:

                st.write(
                    "**Detected:** "
                    + ", ".join(
                        item["detected"]
                    )
                )

            else:

                st.write(
                    "**Detected:** None"
                )


st.divider()


st.caption(
    "ToxicGuard • BERT Multi-label Toxicity Detection"
)