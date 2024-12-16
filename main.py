import os
from modules.data_loader import extract_zip, load_csv
from modules.data_processing import process_gdelt_data
from modules.recommendation import create_faiss_index, recommend_events_faiss

processed_data = None
faiss_index = None
embedding_model = None
indices_map = None

def initialize_backend(zip_file_path, extract_to, text_column="Actor1Name"):
    global processed_data, faiss_index, embedding_model, indices_map
    
    extracted_files = extract_zip(zip_file_path, extract_to)
    if not extracted_files:
        return {"error": "No files were extracted. Please check the ZIP file."}
    
    csv_file = os.path.join(extract_to, extracted_files[0])  
    gdelt_data = load_csv(csv_file)
    if gdelt_data is None or gdelt_data.empty:
        return {"error": f"Failed to load data from {csv_file}."}
    
    processed_data = process_gdelt_data(gdelt_data)
    if processed_data is None or processed_data.empty:
        return {"error": "Data processing failed."}
    
    try:
        faiss_index, embedding_model, indices_map = create_faiss_index(
            processed_data, text_column=text_column
        )
    except Exception as e:
        return {"error": f"Error creating FAISS index: {str(e)}"}
    
    return {"success": True}

def get_recommendations(query, top_k=5, text_column="Actor1Name"):
    global processed_data, faiss_index, embedding_model, indices_map

    if (
        processed_data is None 
        or processed_data.empty 
        or faiss_index is None 
        or embedding_model is None 
        or not indices_map
    ):
        raise ValueError("The backend has not been properly initialized. Ensure all components are loaded.")

    recommendations = recommend_events_faiss(
        processed_data,
        query,
        faiss_index,
        embedding_model,
        indices_map,
        text_column=text_column,
        top_k=top_k
    )
    return recommendations

