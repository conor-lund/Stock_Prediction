"""
Custom transformer classes for the LendingClub Loan Classification project.

These classes are designed to be used inside a scikit-learn / imblearn Pipeline
so that all data cleaning and feature engineering happens within the pipeline
itself (no data leakage). The same Pipeline is later persisted with joblib,
uploaded to S3, and deployed behind a SageMaker endpoint.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


# ---------------------------------------------------------------------------
# 1. Loan-specific cleaning transformers
# ---------------------------------------------------------------------------

class LoanColumnCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans LendingClub raw columns whose strings need to be coerced into
    numeric values (term, emp_length, percentages, dates).

    Each transformation is a separate "step" conceptually, but they are
    grouped here for clarity and to keep the pipeline tidy.
    """

    def __init__(self,
                 term_col='term',
                 emp_length_col='emp_length',
                 percent_cols=('int_rate', 'revol_util'),
                 date_cols=('issue_d', 'earliest_cr_line')):
        # NOTE: sklearn.clone() requires that every __init__ parameter is
        # stored as an attribute with the EXACT SAME value (no list() casts,
        # no copies, no transformations).  We do the iteration-friendly
        # conversion lazily inside transform() instead.
        self.term_col = term_col
        self.emp_length_col = emp_length_col
        self.percent_cols = percent_cols
        self.date_cols = date_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Term: " 36 months" -> 36
        if self.term_col in X.columns:
            X[self.term_col] = (
                X[self.term_col].astype(str)
                                .str.replace('months', '', regex=False)
                                .str.strip()
            )
            X[self.term_col] = pd.to_numeric(X[self.term_col], errors='coerce')

        # Employment length: "10+ years" -> 10, "< 1 year" -> 0
        if self.emp_length_col in X.columns:
            s = (X[self.emp_length_col].astype(str)
                                       .str.replace('10+ years', '10', regex=False)
                                       .str.replace('< 1 year', '0', regex=False))
            X[self.emp_length_col] = pd.to_numeric(s.str.split().str[0],
                                                   errors='coerce')

        # Percent strings: "13.99%" -> 13.99
        for col in list(self.percent_cols or []):
            if col in X.columns:
                X[col] = (X[col].astype(str)
                                .str.replace('%', '', regex=False)
                                .str.strip())
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Date strings like "Dec-2015" -> year as integer (and keep month index)
        for col in list(self.date_cols or []):
            if col in X.columns:
                dt = pd.to_datetime(X[col], format='%b-%Y', errors='coerce')
                X[col + '_year'] = dt.dt.year
                X[col + '_month'] = dt.dt.month
                X = X.drop(columns=[col])

        return X


class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drop columns with a missing-value ratio above the threshold."""

    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.cols_to_keep_ = []

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        ratios = X.isnull().mean()
        self.cols_to_keep_ = ratios[ratios <= self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        keep = [c for c in self.cols_to_keep_ if c in X.columns]
        return X[keep]


class LowVarianceDropper(BaseEstimator, TransformerMixin):
    """Drop columns where the mode accounts for >= dominance_threshold of values."""

    def __init__(self, dominance_threshold=0.95):
        self.dominance_threshold = dominance_threshold
        self.cols_to_keep_ = []

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        keep = []
        for col in X.columns:
            top_freq = X[col].value_counts(dropna=False, normalize=True).iloc[0] \
                if X[col].notna().any() else 1.0
            if top_freq < self.dominance_threshold:
                keep.append(col)
        self.cols_to_keep_ = keep
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        keep = [c for c in self.cols_to_keep_ if c in X.columns]
        return X[keep]


class HighCardinalityDropper(BaseEstimator, TransformerMixin):
    """Drop categorical columns whose cardinality ratio exceeds threshold."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cat = X.select_dtypes(include=['object', 'category']).columns
        n = max(len(X), 1)
        self.cols_to_drop_ = [c for c in cat if X[c].nunique() / n > self.threshold]
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        keep = [c for c in X.columns if c not in self.cols_to_drop_]
        return X[keep]


# ---------------------------------------------------------------------------
# 2. Feature engineering transformers
# ---------------------------------------------------------------------------

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creative engineering for LendingClub:
        - debt-to-income style ratios
        - installment-to-income ratio
        - average FICO
        - credit history length (years between earliest_cr_line_year and issue_d_year)
        - revolving utilization buckets
        - log transforms on heavily skewed monetary columns
    """

    def __init__(self):
        self.applied_log_cols_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1. installment-to-income (monthly)
        if {'installment', 'annual_inc'}.issubset(X.columns):
            monthly_inc = X['annual_inc'].replace(0, np.nan) / 12.0
            X['installment_to_income'] = X['installment'] / monthly_inc

        # 2. loan-to-income
        if {'loan_amnt', 'annual_inc'}.issubset(X.columns):
            X['loan_to_income'] = X['loan_amnt'] / X['annual_inc'].replace(0, np.nan)

        # 3. average FICO
        if {'fico_range_low', 'fico_range_high'}.issubset(X.columns):
            X['fico_avg'] = (X['fico_range_low'] + X['fico_range_high']) / 2.0

        # 4. credit history length (years)
        if {'issue_d_year', 'earliest_cr_line_year'}.issubset(X.columns):
            X['credit_history_yrs'] = X['issue_d_year'] - X['earliest_cr_line_year']

        # 5. revolving utilization bucket (very low / low / med / high)
        if 'revol_util' in X.columns:
            X['revol_util_bucket'] = pd.cut(
                X['revol_util'],
                bins=[-0.1, 25, 50, 75, 200],
                labels=['low', 'med', 'high', 'very_high']
            ).astype(str)

        # 6. log transforms on heavy-tailed monetary features
        for col in ['annual_inc', 'loan_amnt', 'revol_bal']:
            if col in X.columns and X[col].dropna().min() >= 0:
                X['log_' + col] = np.log1p(X[col])

        return X


class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    """
    Apply Yeo-Johnson power transform automatically to numeric columns whose
    absolute skewness exceeds the threshold.
    """

    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self
        skewness = numeric_df.apply(lambda x: skew(x.dropna()) if x.dropna().size else 0.0)
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()
        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols].fillna(0))
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.skewed_cols:
            existing = [c for c in self.skewed_cols if c in X.columns]
            if existing:
                X[existing] = self.pt.transform(X[existing].fillna(0))
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Final selection step:
        - keep numeric columns whose absolute correlation with y is >= corr_threshold
        - keep categorical columns whose cardinality ratio is reasonable
    """

    def __init__(self, missing_threshold=0.3, corr_threshold=0.02,
                 cardinality_threshold=0.9):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.cardinality_threshold = cardinality_threshold
        self.features_to_keep = []

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]

        cat_cols = X_filtered.select_dtypes(exclude='number').columns
        cols_to_drop = []
        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / max(len(X_filtered), 1)
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)
        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]

        numeric_X = X_filtered.select_dtypes(include='number')
        if y is not None and not numeric_X.empty:
            temp_df = numeric_X.copy()
            temp_df['target'] = np.asarray(y)
            correlations = temp_df.corr()['target'].abs().drop('target')
            numeric_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_to_keep = numeric_X.columns.tolist()

        self.features_to_keep = numeric_to_keep + remaining_cats
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        keep = [c for c in self.features_to_keep if c in X.columns]
        return X[keep]
