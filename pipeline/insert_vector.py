import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

print("Loading cleaned data...")
df = pd.read_parquet("data/products_clean.parquet")
embeddings = np.load("data/embeddings.npy")
df = df.reset_index(drop=True)

print(f"Loaded {len(df)} products and {len(embeddings)} embeddings")

print("\nStarting Qdrant locally...")
client = QdrantClient(path="data/qdrant_storage")

print("Creating collection...")
if client.collection_exists("products"):
    client.delete_collection("products")

client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

print("Inserting data into Qdrant...")
points = []
for idx, row in df.iterrows():
    point = PointStruct(
        id=int(idx),
        vector=embeddings[idx].tolist(),
        payload={
            "pid": row["pid"],
            "title": row["title"],
            "brand": row["brand"],
            "category": row["category"],
            "sub_category": row["sub_category"],
            "price": float(row["selling_price"]) if pd.notna(row["selling_price"]) else 0,
            "rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else 0,
        }
    )
    points.append(point)

client.upsert(collection_name="products", points=points)
print(f"\nInserted {len(points)} products into Qdrant ✅")