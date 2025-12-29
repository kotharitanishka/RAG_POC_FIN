from main import create_vectorizer


def test_similarity(query1: str, query2: str):
    
    hf = create_vectorizer()
    """Check distance between two queries."""
    emb1 = hf.embed(query1)
    emb2 = hf.embed(query2)
    
    # Cosine distance
    from numpy import dot
    from numpy.linalg import norm
    similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    distance = 1 - similarity
    
    print(f"Query 1: {query1}")
    print(f"Query 2: {query2}")
    print(f"Distance: {distance:.4f}")
    print(f"Would match at threshold 0.2? {'YES' if distance <= 0.2 else 'NO'}")

# Test it
test_similarity("What tech stacks are used?", "What technologies are used?")
test_similarity("What tech stacks are used?", "What is the project budget?")
test_similarity("What tech stacks are used?", "What are the methodologies used?")
test_similarity("What technologies are used?", "What are the methodologies used?")
test_similarity("What technologies are used?", "What are the methodologies used?")
test_similarity("pranav ne kiski padhai ki hai", "ojas kiske liye padhai kar raha hai")

