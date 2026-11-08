"""Load the Habermas Machine candidate-comparison dataset.

The parquet lives in data/habermas_machine/ (gitignored) and is downloaded on
first use. The full frame is 39,955 rows x 104 columns and expands to ~5 GB in
memory, so pass `columns=` unless you really need all of it.

    from load_habermas import load
    df = load(columns=["question.text", "own_opinion.text", "ratings.agreement"])
"""

import pathlib
import urllib.request

import pandas as pd

URL = "https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet"
PATH = pathlib.Path(__file__).parent / "data" / "habermas_machine" / "hm_all_candidate_comparisons.parquet"


def fetch() -> pathlib.Path:
    """Download the parquet if it isn't already on disk. Returns its path."""
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PATH.with_suffix(".parquet.part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(PATH)  # only claim the real name once the download completed
    return PATH


def load(columns=None) -> pd.DataFrame:
    return pd.read_parquet(fetch(), columns=columns)


if __name__ == "__main__":
    df = load(columns=["question.text", "question.topic", "own_opinion.text"])
    print(df.shape)
    print(df.head())
