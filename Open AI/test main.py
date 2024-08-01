from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_pdf():
    with open("sample.pdf", "rb") as file:
        response = client.post("/upload_pdf/", files={"file": ("sample.pdf", file, "application/pdf")})
        assert response.status_code == 200
        assert "content" in response.json()

def test_ask_question():
    content = "This is a sample document content."
    question = "What is this document about?"
    response = client.post("/ask/", json={"content": content, "question": question})
    assert response.status_code == 200
    assert "answer" in response.json()
