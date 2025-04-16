from sentence_transformers import SentenceTransformer

# Initialize and load the embedding model
# all-MiniLM-L6-v2 is a lightweight model good for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    """
    Convert a list of texts into sentence embeddings.

    Args:
        texts (List[str]): A list of strings (resumes or job descriptions).

    Returns:
        List[np.ndarray]: A list of embedding vectors.
    """
    return model.encode(texts, convert_to_tensor=True)
