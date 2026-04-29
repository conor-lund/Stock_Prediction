"""
Helper utilities for the LendingClub Loan Classification project.

These functions are imported by the main notebook (and by the Streamlit app /
SageMaker inference handler when needed) to load and lightly pre-process the
LendingClub data before it is fed to the cleaning + feature-engineering
pipeline defined in Custom_Classes.py.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DEFAULT_TARGET_MAP = {
    'Fully Paid': 0,
    'Charged Off': 1,
}


def load_loan_data(csv_path: str,
                   target_col: str = 'loan_status',
                   keep_statuses=('Fully Paid', 'Charged Off')) -> pd.DataFrame:
    """
    Load the accepted-loans CSV, keep only Fully Paid / Charged Off rows,
    and return a DataFrame with a binary 'target' column (1 = Charged Off).
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop the unnamed index column that pandas writes back when dataframes
    # are persisted via to_csv() with default settings.
    drop_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df[df[target_col].isin(keep_statuses)].copy()
    df['target'] = df[target_col].map(DEFAULT_TARGET_MAP).astype(int)
    df = df.drop(columns=[target_col])
    df = df.reset_index(drop=True)
    return df


def load_rejected_data(csv_path: str) -> pd.DataFrame:
    """Optional helper if the user wants to compare/contrast rejected loans."""
    df = pd.read_csv(csv_path, low_memory=False)
    drop_cols = [c for c in df.columns if c.startswith('Unnamed')]
    return df.drop(columns=drop_cols) if drop_cols else df


# ---------------------------------------------------------------------------
# Lightweight string-to-number helpers (used both inside the pipeline and
# when the user wants to inspect the data manually).
# ---------------------------------------------------------------------------

def parse_term(value) -> float:
    """' 36 months' -> 36.0"""
    if pd.isna(value):
        return np.nan
    return pd.to_numeric(str(value).replace('months', '').strip(), errors='coerce')


def parse_emp_length(value) -> float:
    """'10+ years' -> 10, '< 1 year' -> 0, '5 years' -> 5"""
    if pd.isna(value):
        return np.nan
    s = (str(value).replace('10+ years', '10')
                  .replace('< 1 year', '0'))
    return pd.to_numeric(s.split()[0] if s.split() else np.nan, errors='coerce')


def parse_percent(value) -> float:
    """'13.99%' -> 13.99"""
    if pd.isna(value):
        return np.nan
    return pd.to_numeric(str(value).replace('%', '').strip(), errors='coerce')


def parse_year_from_date(value) -> float:
    """'Dec-2015' -> 2015"""
    if pd.isna(value):
        return np.nan
    dt = pd.to_datetime(value, format='%b-%Y', errors='coerce')
    return float(dt.year) if pd.notna(dt) else np.nan


# ---------------------------------------------------------------------------
# Column groupings used throughout the project.  Keeping them in one place
# makes it easy to swap them out without touching the notebook.
# ---------------------------------------------------------------------------

# Columns that leak the answer (post-issuance information). Drop before fitting.
LEAKY_COLUMNS = [
    'id', 'member_id', 'url', 'desc', 'title', 'zip_code',
    'out_prncp', 'out_prncp_inv',
    'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt',
    'next_pymnt_d', 'last_credit_pull_d',
    'last_fico_range_high', 'last_fico_range_low',
    'debt_settlement_flag', 'debt_settlement_flag_date',
    'settlement_status', 'settlement_date',
    'settlement_amount', 'settlement_percentage', 'settlement_term',
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
    'deferral_term', 'hardship_amount', 'hardship_start_date',
    'hardship_end_date', 'payment_plan_start_date', 'hardship_length',
    'hardship_dpd', 'hardship_loan_status',
    'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
]


# A compact set of columns the Streamlit app exposes as user inputs.
STREAMLIT_INPUT_FEATURES = [
    'loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_avg',
]


def file_exists(path: str) -> bool:
    return os.path.isfile(path)
