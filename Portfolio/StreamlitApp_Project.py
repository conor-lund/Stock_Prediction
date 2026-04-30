"""
Streamlit UI for the LendingClub Loan Default classifier.

Workflow:
    1. User enters a small set of loan attributes (loan amount, interest rate,
       income, FICO, DTI).
    2. We build a single-row DataFrame whose schema matches X_train.csv (the
       reference training frame written by the notebook).
    3. The pipeline persisted in S3 is invoked via the SageMaker endpoint to
       return a Fully Paid / Charged Off prediction.
    4. The SHAP explainer (also persisted in S3) is loaded and used to render
       a waterfall plot showing which features drove the prediction.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer


from sklearn.pipeline import Pipeline
import shap

from joblib import dump
from joblib import load


# ---------------------------------------------------------------------------
# Setup & path configuration
# ---------------------------------------------------------------------------
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
# Repo layout:
#   <repo_root>/
#       src/                         <- Custom_Classes.py, feature_utils.py
#       Portfolio/
#           StreamlitApp_Project.py  <- this file
#           requirements.txt
current_dir = os.path.dirname(os.path.abspath(__file__))         # .../Portfolio
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # .../<repo_root>
if project_root not in sys.path:
    sys.path.append(project_root)

# src/ is a SIBLING of Portfolio/, so it lives at <repo_root>/src — make sure
# the unpickler can find LoanColumnCleaner / LoanFeatureEngineer / etc.
local_src = os.path.join(project_root, 'src')
if local_src not in sys.path:
    sys.path.append(local_src)

#from src.feature_utils import load_loan_data
#from src.Custom_Classes import LoanColumnCleaner, LoanFeatureEngineer

# Reference training frame used to construct a complete row to send to the
# endpoint.  X_train.csv sits next to this script inside Portfolio/.
file_path = os.path.join(current_dir, 'X_train.csv')

dataset = pd.read_csv(file_path)
dataset = dataset.drop(['Unnamed: 0'], axis=1)
#dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# ---------------------------------------------------------------------------
# AWS credentials are stored in Streamlit Cloud secrets
# ---------------------------------------------------------------------------
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


# AWS Session Management
@st.cache_resource  # avoid re-creating the session on every interaction
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )


session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# ---------------------------------------------------------------------------
# Data & model configuration
# ---------------------------------------------------------------------------
MODEL_INFO = {
    "endpoint" : aws_endpoint,
    "explainer": "explainer_loan.shap",
    "pipeline" : "finalized_loan_model.tar.gz",
    "keys"     : ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_avg'],
    "inputs"   : [
        {"name": "loan_amnt",  "type": "number", "min": 500.0,    "max": 40000.0,   "default": 10000.0,  "step": 100.0},
        {"name": "int_rate",   "type": "number", "min": 5.0,      "max": 35.0,      "default": 12.0,     "step": 0.1},
        {"name": "annual_inc", "type": "number", "min": 5000.0,   "max": 500000.0,  "default": 65000.0,  "step": 500.0},
        {"name": "dti",        "type": "number", "min": 0.0,      "max": 50.0,      "default": 18.0,     "step": 0.1},
        {"name": "fico_avg",   "type": "number", "min": 600.0,    "max": 850.0,     "default": 700.0,    "step": 1.0},
    ],
}


def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}")
    # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    # Load the full pipeline
    return joblib.load(f"{joblib_file}")


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    local_path = local_path

    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def call_model_api(input_df):

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping = {0: "Fully Paid", 1: "Charged Off"}
        return mapping.get(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ---------------------------------------------------------------------------
# Local explainability
# ---------------------------------------------------------------------------
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df = pd.DataFrame(input_df)
    input_df_transformed = preprocessing_pipeline.transform(input_df)

    dataset_1 = dataset.iloc[:, 0:]
    feature_names = dataset_1.columns[1:]
    selector = best_pipeline.named_steps['selector']
    if hasattr(selector, 'get_support'):
        selected_features = feature_names[selector.get_support()]
    else:
        selected_features = pd.Index(getattr(selector, 'features_to_keep', feature_names))

    input_df_transformed = pd.DataFrame(input_df_transformed, columns=selected_features[:input_df_transformed.shape[1]])

    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    try:
        # Tree models return a (n, p, 2) tensor — class 1 is "Charged Off"
        shap.plots.waterfall(shap_values[0, :, 1])
    except Exception:
        shap.plots.waterfall(shap_values[0])
    st.pyplot(fig)

    try:
        top_feature = pd.Series(
            shap_values[0, :, 1].values,
            index=shap_values[0, :, 1].feature_names
        ).abs().idxmax()
    except Exception:
        top_feature = pd.Series(
            shap_values[0].values,
            index=shap_values[0].feature_names
        ).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("👨‍💻 Loan Default Predictor — LendingClub")

with st.form("pred_form"):
    st.subheader("Loan Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'],
                value=inp['default'], step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

# Build a complete row by starting from dataset.iloc[0] and overwriting only the
# user-controlled fields.  This keeps the pipeline happy (it expects the full
# schema it was trained on).
#
# IMPORTANT: use orient='list' so every value is a list (single-element).  If we
# leave the default orient (dict-of-dict for dataset columns) and then update with
# user inputs as lists, pandas barfs with
#     "Mixing dicts with non-Series may lead to ambiguous ordering."
original = dataset.iloc[0:1].to_dict(orient='list')
original.update({k: [v] for k, v in user_inputs.items()})

if submitted:
    res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
