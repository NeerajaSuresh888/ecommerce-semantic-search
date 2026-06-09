import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time

#Load cleaned data
print("Loading cleaned data...")
df = pd.read_parquet("data/products_clean.parquet")
print(f"loaded{len(df)} products")

#load model

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

print("\nGnerating Embedings")

start_time = time.time()

embeddings = model.encode(df["embed_text"].tolist(), batch_size =64, show_progress_bar=True,convert_to_numpy=True)

elapsed =time.time()-start_time
print(f"Generated embeddings for {len(df)} products in {elapsed:.2f} seconds")

#save embeddings
np.save("data/product_embeddings.npy", embeddings)
print(f"Saved embeddings to data/product_embeddings.npy")

print("\n Sanity check - first product: ")
print(f"Product: {df['embed_text'].iloc[0][:60]}...")
print(f"Embedding (first 5 values): {embeddings[0][:5]}")