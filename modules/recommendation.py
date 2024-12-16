from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

def create_faiss_index(data, text_column, model_name='all-MiniLM-L6-v2'):
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
    print("Searching FAISS index...")
    query_vector = model.encode(query).reshape(1, -1).astype('float32')
    distances, faiss_indices = index.search(query_vector, top_k)
    results = [(indices_map[idx], dist) for idx, dist in zip(faiss_indices[0], distances[0])]
    return results

def recommend_events_faiss(data, query, index, model, indices_map, text_column, top_k=5):
    results = search_faiss_index(query, index, model, indices_map, top_k)
    recommended_indices = [row_index for row_index, _ in results]
    recommendations = data.loc[recommended_indices].copy()
    recommendations['similarity_score'] = [1 - dist for _, dist in results]
    return recommendations
