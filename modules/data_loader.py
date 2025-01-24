import requests

def fetch_news_from_api(api_key, query, language='en', page_size=100):
    """
    Fetch news articles from TheNEWSapi based on the query.
    """
    url = "https://api.thenewsapi.com/v1/news/all"
    params = {
        "api_token": api_key,
        "search": query,
        "language": language,
        "page_size": page_size
    }
    try:
        print(f"Fetching news with query: {query}, language: {language}, page_size: {page_size}")
        response = requests.get(url, params=params)
        response.raise_for_status()

        print("API Response:", response.json())

        news_data = response.json()
        articles = news_data.get("data", [])
        print(f"Number of articles fetched: {len(articles)}")
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from API: {e}")
        return [] 
