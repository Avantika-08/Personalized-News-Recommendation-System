import streamlit as st
from main import initialize_backend, get_recommendations

ZIP_FILE_PATH = "data/20150218230000.export.CSV.zip"
EXTRACT_TO = "data"
TEXT_COLUMN = "Actor1Name" 
query = st.text_input("Enter a query:")
top_k = st.number_input("Number of recommendations:", min_value=1, step=1)

if "backend_initialized" not in st.session_state:
    try:
        initialize_backend(ZIP_FILE_PATH, EXTRACT_TO, TEXT_COLUMN)
        st.session_state["backend_initialized"] = True
    except Exception as e:
        st.error(f"Failed to initialize backend: {e}")
        st.stop()

if st.button("Get Recommendations"):
    try:
        recommendations = get_recommendations(query, top_k=top_k, text_column=TEXT_COLUMN)

        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.subheader("Top Recommendations:")
            for _, row in recommendations.iterrows():
                url = row["SOURCEURL"]
                st.markdown(f"[{url}]({url})", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
