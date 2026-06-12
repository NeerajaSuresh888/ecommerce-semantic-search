import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(path="data/qdrant_storage")

def search(query, top_k=5):
    # Embed the query
    query_vector = model.encode(query).tolist()

    #search in Qdrant
    result = client.query_points(
        collection_name = "products",
        query = query_vector,
        limit = top_k

    ).points

    print(f"\nQuery: '{query}'")
    print("-" * 50)
    for i, r in enumerate(result):
        print(f"{i+1}. {r.payload['title']}")
        print(f"   Brand: {r.payload['brand']}")
        print(f"   Price: ₹{r.payload['price']}")
        print(f"   Score: {r.score:.3f}")
        print()

#TEST QUERIES
search("shirts for men")
search("women ethnic wear")
search("casual summer outfit")