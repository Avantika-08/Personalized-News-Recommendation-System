from modules.data_loader import fetch_news_from_api
from modules.recommendation import create_faiss_index, recommend_events_faiss
import pandas as pd
from dotenv import load_dotenv
import os

processed_data = None
faiss_index = None
embedding_model = None
indices_map = None

load_dotenv()
api_key = os.getenv("API_KEY")

def initialize_backend(query, text_column="title", language="en"):
    """
    Initialize the backend by fetching news and creating FAISS index.
    """
    global processed_data, faiss_index, embedding_model, indices_map

    news_articles = fetch_news_from_api(API_KEY, query, language=language)
    if not news_articles:
        return {"error": "Failed to fetch news articles from TheNEWSapi."}
    
    data = pd.DataFrame(news_articles)
    if data.empty:
        return {"error": "No news articles found."}

    if text_column not in data.columns:
        return {"error": f"Column '{text_column}' not found in API response."}

    processed_data = data

    try:
        faiss_index, embedding_model, indices_map = create_faiss_index(
            processed_data, text_column=text_column
        )
    except Exception as e:
        return {"error": f"Error creating FAISS index: {str(e)}"}
    
    return {"success": True}

def get_recommendations(query, top_k=5, text_column="title"):
    global processed_data, faiss_index, embedding_model, indices_map

    if (
        processed_data is None 
        or processed_data.empty 
        or faiss_index is None 
        or embedding_model is None 
        or not indices_map
    ):
        raise ValueError("The backend has not been properly initialized.")

    total_available = len(processed_data)
    print("Total Available:", total_available)

    top_k = min(top_k, total_available)

    try:
        recommendations = recommend_events_faiss(
            processed_data,
            query,
            faiss_index,
            embedding_model,
            indices_map,
            text_column=text_column,
            top_k=top_k
        )

        if recommendations.empty:
            print(f"No recommendations found for query: {query}")
        else:
            print(f"Number of recommendations found: {len(recommendations)}")
        return recommendations

    except Exception as e:
        print(f"Error during recommendation generation: {e}")
        return pd.DataFrame()  
