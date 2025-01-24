from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

def create_faiss_index(data, text_column, model_name='all-MiniLM-L6-v2'):
    """
    Create a FAISS index from the news data.
    """
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings...")
    embeddings = data[text_column].apply(
        lambda x: model.encode(str(x)) if isinstance(x, str) else np.zeros(model.get_sentence_embedding_dimension())
    )
    embeddings_array = np.vstack(embeddings.values).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    print(f"FAISS index created with {index.ntotal} vectors.")
    return index, model, data.index.to_list()

def search_faiss_index(query, index, model, indices_map, top_k=5):
    """
    Search the FAISS index for the most relevant articles.
    """
    query_vector = model.encode(query).reshape(1, -1).astype('float32')
    distances, faiss_indices = index.search(query_vector, top_k)
    results = [(indices_map[idx], dist) for idx, dist in zip(faiss_indices[0], distances[0])]
    return results

def recommend_events_faiss(processed_data, query, faiss_index, embedding_model, indices_map, text_column="title", top_k=5):
    query_embedding = embedding_model.encode(query)

    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)

    indices = indices.flatten()
    distances = distances.flatten()
    
    print(f"Raw FAISS indices: {indices}")
    print(f"Raw FAISS distances: {distances}")

    valid_indices = indices[indices >= 0]
    valid_distances = distances[:len(valid_indices)]
    
    print(f"Valid indices: {valid_indices}")
    print(f"Valid distances: {valid_distances}")

    recommended_articles = []
    for idx in valid_indices:
        row_index = indices_map[idx]
        recommended_articles.append(processed_data.iloc[row_index])

    recommendations = pd.DataFrame(recommended_articles)
    print(f"Recommendations before dropping duplicates: {len(recommendations)}")

    recommendations = recommendations.drop_duplicates(subset=["url"], keep="first")

    print(f"Final recommendations count: {len(recommendations)}")
    return recommendations
