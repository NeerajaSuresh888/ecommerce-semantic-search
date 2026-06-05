import pandas as pd

df = pd.read_csv("data/products.csv")

print("Shape:", df.shape)         
print("\nFirst look:")
print(df.head())                

print("\nColumn info:")
print(df.info())   
               
print("\nMissing values:")
print(df.isnull().sum())          


# remove rows where title or description is missing
df = df.dropna(subset=["title", "description"])


df["title"] = df["title"].str.strip()
df["description"] = df["description"].str.strip()

df = df[df["title"].str.len() > 0]


df["title"] = df["title"].str.lower()
df["description"] = df["description"].str.lower()

df["embed_text"] = df["title"] + ". " + df["description"]

print("\nAfter cleaning:")
print(f"Rows remaining: {len(df)}")
print("\nSample embed_text:")
print(df["embed_text"].head())


df.to_parquet("data/products_clean.parquet", index=False)
print("\nSaved to data/products_clean.parquet ✅")