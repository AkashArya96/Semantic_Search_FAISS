# Semantic_Search_FAISS
1.Install the required Python libraries:
2.Download the food dataset (calories.csv) and place it in the project directory.
3.Start the API:
4.The API will start locally at http://localhost:5000. You can make HTTP requests to this endpoint to perform semantic searches for food items.
# Usage
You can use the API to find similar food items based on a query. You can make a GET request to the /semantic_search endpoint with a query parameter.
The API will respond with a JSON object containing the top 20 similar food items based on the provided query.

# Dependencies
1.Flask: A lightweight web framework for building the API.
2.pandas: Used for handling and processing the food dataset.
3.SentenceTransformer: A library for creating sentence embeddings.
4.faiss: A library for efficient similarity search and clustering of dense vectors.
5.requests: Used for making HTTP requests.

# Acknowledgments
Hugging Face Sentence Transformers for providing the sentence embedding model.
FAISS for efficient similarity search capabilities.
