import streamlit as st
from main import initialize_backend, get_recommendations

TEXT_COLUMN = "title"

st.title("Personalized News Recommender")

query = st.text_input("Enter a query to find relevant news:")
language = st.selectbox("Language:", ["en", "hi"], index=0)

if 'backend_initialized' not in st.session_state:
    st.session_state.backend_initialized = False

if st.button("Initialize Backend", key="init_button"):
    init_response = initialize_backend(query, text_column=TEXT_COLUMN, language=language)
    if "error" in init_response:
        st.error(init_response["error"])
    else:
        st.session_state.backend_initialized = True
        st.success("Backend initialized successfully!")

if st.session_state.backend_initialized:
    if st.button("Get Recommendations", key="get_recommendations_button"):
        try:
            recommendations = get_recommendations(query, text_column=TEXT_COLUMN)

            if not recommendations.empty:
                st.subheader("Top Recommendations:")
                for _, row in recommendations.iterrows():
                    url = row["url"]
                    st.markdown(f"[{row['title']}]({url})", unsafe_allow_html=True)
            else:
                st.warning("No recommendations found.")
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
else:
    st.warning("Please initialize the backend first.")
