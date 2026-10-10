from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "date_",
    "futcode",
    "lasttrddate",
    "settlement",
]





def load_raw_data(path):
    """Load contract-level futures data from Parquet."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_parquet(path)


def clean_futures_data(df):
    """Validate dates, remove invalid observations,
    and handle duplicate contract-date records."""
    df = df.copy()

    required = REQUIRED_COLUMNS
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date_"] = pd.to_datetime(df["date_"], errors="coerce")
    df["lasttrddate"] = pd.to_datetime(df["lasttrddate"], errors="coerce")
    df["settlement"] = pd.to_numeric(df["settlement"], errors="coerce")

    df = df.dropna(subset=required)
    df = df[np.isfinite(df["settlement"]) & (df["settlement"] > 0)].copy()

    keys = ["date_", "futcode"]

    
    # same futures contract has different settlement prices 
    # or expiration dates on the same trading day.
    conflicts = (df.groupby(keys)[["settlement", "lasttrddate"]].nunique())

    if (conflicts > 1).any().any():
        raise ValueError("Conflicting duplicate contract-date records.")

    df = df.drop_duplicates(subset=keys)

    return df.sort_values(keys).reset_index(drop=True)

def calculate_maturity(df):
    """Calculate time to maturity in years."""
    df = df.copy()

    df["tau"] = (
        df["lasttrddate"] - df["date_"]).dt.days / 365.25

    df = df[df["tau"] > 0].copy()

    df["log_settlement"] = np.log(df["settlement"])

    return df.reset_index(drop=True)



    
def select_nearby_contracts(df, n=6):
    """Select the first n eligible contracts per date."""
    df = df.copy()

    df = df.sort_values(["date_", "tau", "futcode"])

    df["maturity_rank"] = (df.groupby("date_").cumcount() + 1)

    df = df[df["maturity_rank"] <= n].copy()

    return df.reset_index(drop=True)

def validate_panel(df, n=6):
    """Check contract coverage, maturity ordering,
    settlement prices, and missing observations."""
    required = [
        "date_",
        "futcode",
        "lasttrddate",
        "settlement",
        "tau",
        "log_settlement",
        "maturity_rank",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Panel is empty.")

    if df[required].isna().any().any():
        raise ValueError("Panel contains missing values.")

    if not np.isfinite(df["settlement"]).all():
        raise ValueError("Panel contains nonfinite settlement prices.")

    if (df["settlement"] <= 0).any():
        raise ValueError("Panel contains nonpositive settlement prices.")

    if not np.isfinite(df["tau"]).all() or (df["tau"] <= 0).any():
        raise ValueError("Panel contains invalid maturities.")

    if not np.allclose(
        df["log_settlement"],
        np.log(df["settlement"]),
    ):
        raise ValueError("Log settlement values are inconsistent.")

    if df.duplicated(["date_", "futcode"]).any():
        raise ValueError("Duplicate contract-date observations found.")

    counts = df.groupby("date_")["futcode"].nunique()
    incomplete = counts[counts != n]

    if not incomplete.empty:
        raise ValueError(
            f"{len(incomplete)} dates do not have exactly "
            f"{n} contracts:\n{incomplete.head(10)}"
        )

    for date, group in df.groupby("date_"):
        group = group.sort_values("maturity_rank")

        if not np.array_equal(
            group["maturity_rank"].to_numpy(),
            np.arange(1, n + 1),
        ):
            raise ValueError(f"Invalid maturity ranks on {date}.")

        if not (np.diff(group["tau"].to_numpy()) > 0).all():
            raise ValueError(f"Maturities are not increasing on {date}.")

    return {
        "trading_days": df["date_"].nunique(),
        "observations": len(df),
        "unique_contracts": df["futcode"].nunique(),
        "contracts_per_day": n,
        "min_tau": df["tau"].min(),
        "max_tau": df["tau"].max(),
        "min_settlement": df["settlement"].min(),
        "max_settlement": df["settlement"].max(),
    }



def prepare_panel(df, n=6):
    """Run the complete data preparation pipeline."""
    df = clean_futures_data(df)
    df = calculate_maturity(df)
    df = select_nearby_contracts(df, n=n)
    validate_panel(df, n=n)
    return df




def prepare_kalman_data(df, n=None):
    """Return finite float matrices for a complete, chronological rank panel.

    ``n`` defaults to the panel's number of ranks, so dataset loaders can supply
    different curve sizes. Explicit float conversion handles pandas nullable
    numeric columns, whose default NumPy representation can be object dtype.
    """
    df = df.copy()
    df["date_"] = pd.to_datetime(df["date_"])
    if n is None:
        n = df["maturity_rank"].nunique()
    validate_panel(df, n=n)

    # Convert long-format data to matrices
    prices = df.pivot(
        index="date_",
        columns="maturity_rank",
        values="log_settlement",
    )

    maturities = df.pivot(
        index="date_",
        columns="maturity_rank",
        values="tau",
    )

    # Ensure dates and maturities are sorted
    prices = prices.sort_index().sort_index(axis=1)
    maturities = maturities.reindex(
        index=prices.index,
        columns=prices.columns,
    )

    # Check data completeness
    if prices.shape[1] != n:
        raise ValueError(f"Expected {n} maturity ranks")

    if prices.isna().any().any():
        raise ValueError("Missing futures prices")

    if maturities.isna().any().any():
        raise ValueError("Missing maturities")

    if (maturities.to_numpy(dtype=float) <= 0).any():
        raise ValueError("Maturities must be positive")

    dates = prices.index
    Y = prices.to_numpy(dtype=float)
    tau = maturities.to_numpy(dtype=float)
    if not np.isfinite(Y).all() or not np.isfinite(tau).all():
        raise ValueError("Prices and maturities must be finite")

    # Time between consecutive trading days (calendar years)
    dt = np.diff(dates.to_numpy()) / np.timedelta64(1, "D")
    dt = dt / 365.25

    return dates, Y, tau, dt
