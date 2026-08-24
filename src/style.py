import streamlit as st


def apply_styles():

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"]
        [data-testid="stSidebarNav"] a {
            font-size: 18px;
            font-weight: 700;
        }

        section[data-testid="stSidebar"]
        [data-testid="stSidebarNav"] a span {
            font-size: 18px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )