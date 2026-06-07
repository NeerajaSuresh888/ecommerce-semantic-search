import pandas as pd
import json

print("Loading dataset...")
with open("flipkart_fashion_products_dataset.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
print(f"Shape: {df.shape}")

df = df[["pid", "title", "description", "brand", "category", "sub_category", "selling_price", "average_rating"]]
print(f"After column selection: {df.shape}")

df = df.dropna(subset=["title", "description"])

df["title"] = df["title"].str.strip().str.lower()
df["description"] = df["description"].str.strip().str.lower()
df["brand"] = df["brand"].fillna("unknown").str.strip().str.lower()

df = df[df["title"].str.len() > 0]
df = df[df["description"].str.len() > 0]

df["selling_price"] = pd.to_numeric(df["selling_price"].astype(str).str.replace(",", ""), errors="coerce")
df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")

df["embed_text"] = df["title"] + ". " + df["brand"] + ". " + df["description"]

print(f"Rows after cleaning: {len(df)}")
print("\nSample embed_text:")
print(df["embed_text"].head(3).to_string())

df.to_parquet("data/products_clean.parquet", index=False)
print(f"\nSaved {len(df)} products ")