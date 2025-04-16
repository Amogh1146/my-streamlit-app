from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rank_resumes(job_embedding, resume_embeddings, filenames, top_n=10):
    """
    Rank resumes by cosine similarity to the job description embedding.

    Args:
        job_embedding (np.array): Embedding of job description.
        resume_embeddings (np.array): Embeddings of resumes.
        filenames (List[str]): Filenames of the resumes.
        top_n (int): Number of top resumes to return.

    Returns:
        Tuple[List[dict], List[dict]]: 
            - List of top resumes (filename, similarity_score).
            - List of additional resumes with scores between 50.0% and 51.0%.
    """
    similarity_scores = cosine_similarity(
        job_embedding.reshape(1, -1), resume_embeddings
    )[0]

    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    # Top 10 resumes
    top_resumes = [
        {
            "Resume": filenames[i],
            "Relevance Score": float(similarity_scores[i])
        }
        for i in ranked_indices[:top_n]
    ]
    
    # Additional resumes with score between 50.0% and 51.0%
    additional_resumes = [
        {
            "Resume": filenames[i],
            "Relevance Score": float(similarity_scores[i])
        }
        for i in ranked_indices[top_n:]
        if 50.0 <= similarity_scores[i] < 51.0
    ]

    return top_resumes, additional_resumes
