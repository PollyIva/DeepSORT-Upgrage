"""Helpers for analyzing experiment results."""

import pandas as pd


def load_results(path):
    return pd.read_csv(path)


def filter_by_fps(df, min_fps=5.0):
    return df[df["fps"] >= min_fps].copy()


def best_by_sequence(df, metric="HOTA"):
    return (
        df.sort_values(metric, ascending=False)
        .groupby("sequence")
        .first()
        .reset_index()
    )
