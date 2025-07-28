import redis
import json
import numpy as np
from Embedding import MiniLMEmbedder  # Import MiniLM embedding model
from redis.commands.search.query import Query
import ollama


# Initialize Redis client
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

VECTOR_DIM = 384
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Initialize MiniLM Embedder
embedder = MiniLMEmbedder()

def get_embedding(text):
    """Generate embedding"""
    return embedder.embed_chunks([text])[0]  

def search_embeddings(query, top_k=3):
    """Search the Redis database for similar embeddings."""
    query_embedding = get_embedding(query)
    
    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "text", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into a structured format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.text,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Debugging output
        for result in top_results:
            print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    """Generate a response using the retrieved context and the Ollama Mistral model."""

    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )


    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'. If you feel like you don't have enough information, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="mixtral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Print retrieved context
        print("\n--- Retrieved Context ---")
        for result in context_results:
            print(f"File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        # Generate RAG response using the retrieved context
        response = generate_rag_response(query, context_results)

        # Print the generated response
        print("\n--- Response ---")
        print(response)



if __name__ == "__main__":
    interactive_search()
