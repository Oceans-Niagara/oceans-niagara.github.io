import os
import io
import zipfile
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "orchestrator"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def process_and_upload():
    documents = []
    
    page_url = "https://github.io"
    try:
        page_res = requests.get(page_url)
        if page_res.status_code == 200:
            soup = BeautifulSoup(page_res.text, 'html.parser')
            for el in soup(["script", "style", "nav", "footer"]):
                el.decompose()
            page_text = soup.get_text(separator="\n", strip=True)
            for idx, chunk in enumerate(chunk_text(page_text)):
                documents.append({
                    "id": f"live-arch-page-chunk-{idx}",
                    "text": chunk,
                    "source": "oceans-niagara.github.io/tools/index.html"
                })
    except Exception as e:
        print(f"Warning: {e}")

    zip_url = "https://github.com"
    try:
        response = requests.get(zip_url)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for filename in z.namelist():
                    if filename.endswith('/') or filename.startswith('__MACOSX'):
                        continue 
                    with z.open(filename) as f:
                        try:
                            content = f.read().decode('utf-8')
                        except UnicodeDecodeError:
                            continue
                        for idx, chunk in enumerate(chunk_text(content)):
                            documents.append({
                                "id": f"{filename}-chunk-{idx}",
                                "text": chunk,
                                "source": filename
                            })
    except Exception as e:
        print(f"Warning fetching zip: {e}")

    if not documents:
        return

    texts_to_embed = [doc["text"] for doc in documents]
    embeddings = model.encode(texts_to_embed, show_progress_bar=False)

    vectors_to_upsert = []
    for idx, doc in enumerate(documents):
        vectors_to_upsert.append({
            "id": doc["id"],
            "values": embeddings[idx].tolist(),
            "metadata": {"text": doc["text"], "source": doc["source"]}
        })

    for i in range(0, len(vectors_to_upsert), 50):
        index.upsert(vectors=vectors_to_upsert[i:i+50])

if __name__ == "__main__":
    process_and_upload()

