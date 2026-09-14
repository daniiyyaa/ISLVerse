import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "data/raw/sentence_gloss.csv"

df = pd.read_csv(input_file)

# Use the actual column names from your CSV
df = df[["Sentence", "SIGN GLOSSES"]].dropna()

# Rename columns for easier processing
df = df.rename(columns={
    "Sentence": "english",
    "SIGN GLOSSES": "gloss"
})

# Clean the text
df["english"] = (
    df["english"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["gloss"] = (
    df["gloss"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Remove empty and duplicate rows
df = df[
    (df["english"] != "") &
    (df["gloss"] != "")
]

df = df.drop_duplicates()

# Split: 80% training, 10% validation, 10% testing
train_df, temporary_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

valid_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42
)

train_df.to_csv("data/processed/train.csv", index=False)
valid_df.to_csv("data/processed/valid.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Dataset preparation completed.")
print("Total samples:", len(df))
print("Training samples:", len(train_df))
print("Validation samples:", len(valid_df))
print("Testing samples:", len(test_df))