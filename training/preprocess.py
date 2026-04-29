import argparse
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from pathlib import Path


QOS_PRIORITY = {
    "VOIP":      0,  # top priority
    "STREAMING": 1,
    "CHAT":      2,
    "BROWSING":  3,
    "MAIL":      4,
    "FT":        5,
    "P2P":       6,  
}


def load_arff(path: str) -> pd.DataFrame:
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # modify ARFF from bytes to str
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8").str.strip()

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"[clean] NaN/Inf: {before - len(df)} , remaining {len(df)}")
    return df


def inspect(df: pd.DataFrame, label_col: str):
    print(f"\n[inspect] shape: {df.shape}")
    print(f"[inspect] features: {[c for c in df.columns if c != label_col]}")
    print(f"\n[inspect] label distribution:")
    counts = df[label_col].value_counts()
    for cls, cnt in counts.items():
        print(f"  {cls:<12} {cnt:>6}  ({cnt/len(df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="ARFF path")
    parser.add_argument("--outdir", default="data/processed", help="output folder")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[load] {args.input}")
    df = load_arff(args.input)
    df = clean(df)

    # class1 or class2
    label_col = next(c for c in df.columns if c.startswith("class"))
    print(f"[info] label : {label_col}")

    inspect(df, label_col)

    # just to upper
    df[label_col] = df[label_col].str.upper()

    df["qos_priority"] = df[label_col].map(QOS_PRIORITY)

    feature_cols = [c for c in df.columns if c not in (label_col, "qos_priority")]
    X = df[feature_cols]
    y = df[label_col]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"\n[split] train: {len(X_train)}, test: {len(X_test)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[label_col] = y_train.values
    test_df = X_test.copy()
    test_df[label_col] = y_test.values

    train_path = outdir / "train.csv"
    test_path  = outdir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    feature_path = outdir / "features.txt"
    feature_path.write_text("\n".join(feature_cols))

    print(f"\n[done] output to {outdir}/")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {feature_path}")


if __name__ == "__main__":
    main()