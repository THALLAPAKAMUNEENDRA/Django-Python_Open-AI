from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_content(content: str, query: str) -> str:
    paragraphs = content.split('\n')
    vectorizer = TfidfVectorizer().fit_transform(paragraphs + [query])
    vectors = vectorizer.toarray()
    query_vector = vectors[-1]
    similarities = cosine_similarity([query_vector], vectors[:-1])[0]
    most_similar_index = similarities.argmax()
    return paragraphs[most_similar_index]

@app.post("/ask/")
async def ask_question(content: str, question: str):
    relevant_content = retrieve_relevant_content(content, question)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Relevant Content:\n{relevant_content}\n\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )
    answer = response.choices[0].text.strip()
    return {"answer": answer}
