import streamlit as st
import pandas as pd
import pickle
import requests
import sklearn
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, plot_model as cls_plot, tune_model as cls_tune
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, plot_model as reg_plot, tune_model as reg_tune

from db import create_tables, add_user, get_hashed_password, save_dataset, get_all_users, get_all_datasets
from streamlit_authenticator.utilities.hasher import Hasher

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AutoML Autonomous AI", layout="wide")
create_tables()

# ---------------- UI ----------------
st.title("🚀 Autonomous AutoML AI SaaS")

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
h1, h2, h3 { color: #00ffcc; }
.stButton>button { background-color:#00ffcc; color:black; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
for key in ["logged_in", "username", "df", "model", "mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.session_state["logged_in"] = st.session_state.get("logged_in", False)

# ---------------- LOGIN ----------------
if not st.session_state["logged_in"]:
    choice = st.sidebar.selectbox("Login / Signup", ["Login", "Signup"])

    if choice == "Signup":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Create"):
            add_user(u, Hasher.hash(p))
            st.success("Account created")

    else:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            stored = get_hashed_password(u)
            if stored and Hasher.check_pw(p, stored):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.success("Logged in")
            else:
                st.error("Invalid login")

    st.stop()

# ---------------- MAIN ----------------
username = st.session_state.username
st.sidebar.success(f"Welcome {username}")

menu = st.sidebar.radio("Menu", ["Upload", "EDA", "Train", "Visualize", "Download", "Admin"])

# ---------------- UPLOAD ----------------
if menu == "Upload":
    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        st.session_state.df = df
        save_dataset(username, file.name)

        st.success("Dataset loaded")
        st.dataframe(df.head())

# ---------------- EDA ----------------
elif menu == "EDA":
    df = st.session_state.df

    if df is not None:
        st.write(df.describe())
        st.write(df.info())
        st.write(df.isnull().sum())
        st.write(df.columns)

# ---------------- TRAIN ----------------
elif menu == "Train":
    df = st.session_state.df

    if df is not None:
        target = st.selectbox("Target Column", df.columns)

        mode = "Classification" if df[target].nunique() < 20 else "Regression"
        st.session_state.mode = mode

        if st.button("Run AutoML"):
            if mode == "Classification":
                st.session_state.mode = mode
                st.info(f"Detected: {mode}")
                cls_setup(data=df, target=target, verbose=False)
                best_model = cls_compare()
                model = cls_tune(best_model)
                results = cls_pull()

            elif mode == "Clustering":
                st.session_state.mode = mode
                st.info(f"Detected: {mode}")
                clu_setup(data=df, target=target, verbose=False)
                best_model = clu_compare()
                model = clu_tune(best_model)
                results = clu_pull()

            elif mode == "Anomaly":
                st.session_state.mode = mode
                st.info(f"Detected: {mode}")
                ano_setup(data=df, target=target, verbose=False)
                best_model = ano_compare()
                model = ano_tune(best_model)
                results = ano_pull()

            else:
                st.info(f"Detected: {mode}")
                reg_setup(data=df, target=target, verbose=False)
                best_model = reg_compare()
                model = reg_tune(best_model)
                results = reg_pull()

            st.session_state.model = model
            st.success("Training complete")
            st.write(f"Best Model: {model}")
            st.dataframe(results)

# ---------------- VISUALIZE ----------------
elif menu == "Visualize":
    model = st.session_state.model
    mode = st.session_state.mode

    if model:
        if mode == "Classification":
            viz_option = st.selectbox("Visualization", [
                "Confusion Matrix",
                "Feature Importance",
                "AUC",
                "PR Curve",
                "Learning Curve",
                "Class Report",
                "KS Statistic"
            ])

            try:
                if viz_option == "Confusion Matrix":
                    cls_plot(model, plot="confusion_matrix", display_format="streamlit")
                elif viz_option == "Feature Importance":
                    cls_plot(model, plot="feature", display_format="streamlit")
                elif viz_option == "AUC":
                    cls_plot(model, plot="auc", display_format="streamlit")
                elif viz_option == "PR Curve":
                    cls_plot(model, plot="pr", display_format="streamlit")
                elif viz_option == "Learning Curve":
                    cls_plot(model, plot="learning", display_format="streamlit")
                elif viz_option == "Class Report":
                    cls_plot(model, plot="class_report", display_format="streamlit")
                elif viz_option == "KS Statistic":
                    cls_plot(model, plot="ks", display_format="streamlit")
            except:
                st.warning("Visualization not supported")

        else:
            viz_option = st.selectbox("Visualization", [
                "Feature Importance",
                "Residuals",
                "Prediction Error",
                "Learning Curve"
            ])

            try:
                if viz_option == "Feature Importance":
                    reg_plot(model, plot="feature", display_format="streamlit")
                elif viz_option == "Residuals":
                    reg_plot(model, plot="residuals", display_format="streamlit")
                elif viz_option == "Prediction Error":
                    reg_plot(model, plot="error", display_format="streamlit")
                elif viz_option == "Learning Curve":
                    reg_plot(model, plot="learning", display_format="streamlit")
            except:
                st.warning("Visualization not supported")

# ---------------- DOWNLOAD ----------------
elif menu == "Download":
    model = st.session_state.model

    if model:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open("model.pkl", "rb") as f:
            st.download_button("Download Model", f, file_name="model.pkl")

# ---------------- ADMIN ----------------
elif menu == "Admin":
    st.write("Users")
    st.dataframe(pd.DataFrame(get_all_users(), columns=["ID", "User"]))

    st.write("Datasets")
    st.dataframe(pd.DataFrame(get_all_datasets(), columns=["ID", "User", "File"]))

# =====================================================
# 🤖 AUTONOMOUS AI AGENT (OLLAMA)
# =====================================================

st.markdown("---")
st.markdown("## 🤖 Autonomous AI Data Scientist")

def route_action(text):
    text = text.lower()

    if "train" in text or "best model" in text:
        return "TRAIN"
    if "improve" in text or "accuracy" in text:
        return "IMPROVE"
    if "clean" in text:
        return "CLEAN"

    return "CHAT"


def ai_agent(prompt, df=None, model=None):
    context = ""

    if df is not None:
        context += f"""
Columns: {list(df.columns)}
Shape: {df.shape}
Missing: {df.isnull().sum().to_dict()}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:latest",
            "prompt": f"{context}\nUser: {prompt}",
            "stream": False
        }
    )

    return response.json().get("response", "")


# ---------------- CHAT MEMORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

user_input = st.chat_input("Ask AI: train, improve, analyze...")

if user_input:
    st.session_state.chat.append({"role": "user", "text": user_input})

    df = st.session_state.df
    action = route_action(user_input)

    if action == "TRAIN" and df is not None:
        target = df.columns[-1]

        if df[target].nunique() < 20:
            cls_setup(df, target=target, silent=True)
            best_model = cls_compare()
            model = cls_tune(best_model)
        else:
            reg_setup(df, target=target, silent=True)
            best_model = reg_compare()
            model = reg_tune(best_model)

        st.session_state.model = model
        reply = "🚀 Auto training + tuning completed."

    elif action == "IMPROVE":
        reply = "🧠 Try feature engineering, scaling, and ensemble boosting."

    elif action == "CLEAN":
        reply = "🧹 Missing values already handled automatically."

    else:
        reply = ai_agent(user_input, df, st.session_state.model)

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.chat.append({"role": "assistant", "text": reply})