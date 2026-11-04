import pandas as pd
from sklearn.preprocessing import StandardScaler


def validate_dataframe(df, required_columns, target_column):
    """Check that a dataframe meets basic requirements."""

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if len(df) == 0:
        raise ValueError("Dataframe is empty")

    return True


def clean_data(
    df,
    numeric_columns,
    categorical_columns,
    missing_strategy="median",
):
    """
    Clean a dataframe by handling missing values.

    Numeric columns:
        Missing values are filled with the median.

    Categorical columns:
        Missing values are filled with the mode.

    The original dataframe is not modified.
    """

    df = df.copy()

    if missing_strategy == "drop":
        return df.dropna()

    if missing_strategy != "median":
        raise ValueError(
            f"Unknown missing-value strategy: {missing_strategy}"
        )

    # Fill numeric missing values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if col in df.columns:
            mode_val = df[col].mode()

            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    return df


def encode_categoricals(df, columns):
    """
    One-hot encode categorical columns.

    The original dataframe is not modified.
    """

    df = df.copy()

    df = pd.get_dummies(
        df,
        columns=columns,
        drop_first=True,
        dtype=int,
    )

    return df


def scale_numeric_features(df, numeric_columns):
    """
    Standardize numeric features using StandardScaler.

    The original dataframe is not modified.

    Returns
    -------
    scaled_df : pandas.DataFrame
        Dataframe containing scaled numeric columns.

    scaler : StandardScaler
        Fitted scaler that can be reused on new data.
    """

    df = df.copy()

    scaler = StandardScaler()

    columns_to_scale = [
        col for col in numeric_columns
        if col in df.columns
    ]

    if columns_to_scale:
        df[columns_to_scale] = scaler.fit_transform(
            df[columns_to_scale]
        )

    return df, scaler


def check_data_quality(df, numeric_columns):
    """Return a dictionary of data quality metrics."""

    total_cells = len(df) * len(df.columns)

    report = {
        "total_rows": len(df),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_percentage": round(
            df.isnull().sum().sum() / total_cells * 100,
            2,
        ),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in numeric_columns:
        if col in df.columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())

    return report
