from transformers import pipeline

def summarize_events(recommendations, text_column, chunk_size=500):
    """
    Summarize the text content of recommended events.

    Args:
        recommendations (pd.DataFrame): DataFrame containing recommended events.
        text_column (str): Name of the column containing text to summarize.
        chunk_size (int): Maximum number of tokens per chunk for summarization.

    Returns:
        str: Summary of the text content.
    """

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text_to_summarize = " ".join(recommendations[text_column].astype(str).tolist())
    
    # Split the text into chunks of size `chunk_size`
    chunks = [text_to_summarize[i:i+chunk_size] for i in range(0, len(text_to_summarize), chunk_size)]
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
    
    # Combine all chunk summaries into a single summary
    return " ".join(summaries)
