import streamlit as st
import pandas as pd
import pickle
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, plot_model as cls_plot
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, plot_model as reg_plot
from db import create_tables, add_user, get_hashed_password, save_dataset, get_all_users, get_all_datasets, delete_user, delete_dataset
from streamlit_authenticator.utilities.hasher import Hasher
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AutoML SaaS Pro", layout="wide")
create_tables()

# ---------------- PREMIUM UI ----------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.sidebar .sidebar-content { background-color: #111827; color: white; }
h1, h2, h3 { color: #00ffcc; }
.stButton>button { background-color: #00ffcc; color: black; border-radius: 10px; height: 3em; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 AutoML SaaS Pro")

# ---------------- SESSION STATE ----------------
for key in ["logged_in", "username", "df", "model", "mode"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state["logged_in"] = st.session_state.get("logged_in", False)

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state["logged_in"]:
    auth_choice = st.sidebar.selectbox("Choose", ["Login", "Signup"])

    # -------- SIGNUP --------
    if auth_choice == "Signup":
        st.subheader("📝 Create Account")
        new_user = st.text_input("Username", key="signup_user")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Signup"):
            hashed_password = Hasher.hash(new_password)
            add_user(new_user, hashed_password)
            st.success("✅ Account created! Please login.")

    # -------- LOGIN --------
    elif auth_choice == "Login":
        st.subheader("🔑 Login")
        username_input = st.text_input("Username", key="login_user")
        password_input = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            stored_hashed = get_hashed_password(username_input)
            if stored_hashed and Hasher.check_pw(password_input, stored_hashed):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username_input
                st.success(f"✅ Logged in as {username_input}")
            else:
                st.error("❌ Invalid username or password")
    st.stop()  # Stops execution until user logs in

# ---------------- MAIN APP ----------------
username = st.session_state["username"]
st.sidebar.success(f"Welcome {username}")

# ---------------- SIDEBAR NAVIGATION ----------------
menu_options = ["Upload", "EDA", "Train", "Visualize", "Download"]
if username == "admin":
    menu_options.append("Admin")
menu = st.sidebar.radio("Navigation", menu_options)

# ---------------- UPLOAD ----------------
if menu == "Upload":
    st.header("📂 Upload Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        st.session_state.df = df
        save_dataset(username, file.name)
        st.success(f"✅ Dataset Loaded: {file.name}")
        st.dataframe(df.head())

# ---------------- EDA ----------------
elif menu == "EDA":
    st.header("📊 Data Analysis")
    df = st.session_state.df
    if df is not None:
        st.subheader("Statistics")
        st.write(df.describe())
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        st.subheader("Columns")
        st.write(df.columns.tolist())
    else:
        st.warning("⚠️ Upload dataset first")

# ---------------- TRAIN ----------------
elif menu == "Train":
    st.header("🤖 Train AutoML")
    df = st.session_state.df

    if df is not None:
        # -------- Missing Values Check --------
        if df.isnull().sum().sum() > 0:
            st.warning("⚠️ Missing values detected. They will be automatically filled:")
            for col in df.columns:
                if df[col].dtype in ["float64", "int64"]:
                    df[col].fillna(df[col].mean(), inplace=True)
                    st.info(f"Filled missing numerical values in '{col}' with mean")
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    st.info(f"Filled missing categorical values in '{col}' with mode")
        else:
            st.info("✅ No missing values detected.")

        # -------- Column Type Detection --------
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        st.info(f"Numerical columns: {num_cols}")
        st.info(f"Categorical columns: {cat_cols}")

        # -------- Target Selection & Auto Task Detection --------
        target = st.selectbox("🎯 Select Target Column", df.columns)
        mode = "Classification" if df[target].nunique() < 10 else "Regression"
        st.info(f"Detected task: {mode}")
        st.session_state.mode = mode

        # -------- Run AutoML --------
        if st.button("🚀 Run AutoML"):
            try:
                if mode == "Classification":
                    cls_setup(data=df, target=target, session_id=123, html=False)
                    model = cls_compare()
                    results = cls_pull()
                else:
                    reg_setup(data=df, target=target, session_id=123, html=False)
                    model = reg_compare()
                    results = reg_pull()

                st.session_state.model = model
                st.success("✅ Training Complete")
                st.subheader("🏆 Leaderboard")
                st.dataframe(results)
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Upload dataset first")

# ---------------- VISUALIZE ----------------
elif menu == "Visualize":
    st.header("📊 Model Visualizations")
    model = st.session_state.model
    mode = st.session_state.mode
    if model is not None:
        if mode == "Classification":
            viz_option = st.selectbox("Choose Visualization", [
                "Confusion Matrix", "Feature Importance", "AUC Curve",
                "Precision-Recall Curve", "Learning Curve", "Residuals"
            ])
            try:
                plot_map = {
                    "Confusion Matrix": "confusion_matrix",
                    "Feature Importance": "feature",
                    "AUC Curve": "auc",
                    "Precision-Recall Curve": "pr",
                    "Learning Curve": "learning",
                    "Residuals": "residuals"
                }
                cls_plot(model, plot=plot_map[viz_option], display_format="streamlit")
            except Exception:
                st.warning("⚠️ Visualization not supported")
        else:  # Regression
            viz_option = st.selectbox("Choose Visualization", [
                "Feature Importance", "Residuals", "Prediction Error", "Learning Curve", "Cook's Distance"
            ])
            try:
                plot_map = {
                    "Feature Importance": "feature",
                    "Residuals": "residuals",
                    "Prediction Error": "error",
                    "Learning Curve": "learning",
                    "Cook's Distance": "cooks"
                }
                reg_plot(model, plot=plot_map[viz_option], display_format="streamlit")
            except Exception:
                st.warning("⚠️ Visualization not supported")
    else:
        st.warning("⚠️ Train model first")

# ---------------- DOWNLOAD ----------------
elif menu == "Download":
    st.header("📥 Download Model")
    model = st.session_state.model
    if model is not None:
        with open("best_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, file_name="best_model.pkl")
    else:
        st.warning("⚠️ Train model first")

# ---------------- ADMIN ----------------
elif menu == "Admin":
    st.header("🛠 Admin Dashboard")
    # Users
    st.subheader("All Users")
    users = get_all_users()
    if users:
        users_df = pd.DataFrame(users, columns=["ID", "Username"])
        st.dataframe(users_df)
        del_user_id = st.number_input("Enter User ID to Delete", min_value=0, step=1)
        if st.button("Delete User"):
            delete_user(del_user_id)
            st.success(f"✅ User {del_user_id} deleted!")
            st.experimental_rerun()
    else:
        st.info("No users yet.")

    # Datasets
    st.subheader("All Datasets")
    datasets = get_all_datasets()
    if datasets:
        datasets_df = pd.DataFrame(datasets, columns=["ID", "Username", "Filename"])
        st.dataframe(datasets_df)
        del_dataset_id = st.number_input("Enter Dataset ID to Delete", min_value=0, step=1)
        if st.button("Delete Dataset"):
            delete_dataset(del_dataset_id)
            st.success(f"✅ Dataset {del_dataset_id} deleted!")
            st.experimental_rerun()
    else:
        st.info("No datasets yet.")