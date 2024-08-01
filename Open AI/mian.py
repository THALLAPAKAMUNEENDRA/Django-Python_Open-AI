from fastapi import FastAPI, UploadFile, File, HTTPException
from PyPDF2 import PdfFileReader
import openai
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Replace with your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    pdf_reader = PdfFileReader(io.BytesIO(await file.read()))
    text_content = ""
    
    for page_num in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_num)
        text_content += page.extractText()
    
    return {"content": text_content}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
