import os
import io
import time
import zipfile
import requests
from bs4 import BeautifulSoup
from google import genai
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "orchestrator"))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def process_and_upload():
    documents = []
    
    page_url = "https://oceans-niagara.github.io/tools/index.html"
    print(f"Scraping live architecture page: {page_url}")
    try:
        page_res = requests.get(page_url)
        if page_res.status_code == 200:
            soup = BeautifulSoup(page_res.text, 'html.parser')
            for script_or_style in soup(["script", "style", "nav", "footer"]):
                script_or_style.decompose()
            page_text = soup.get_text(separator="\n", strip=True)
            
            for idx, chunk in enumerate(chunk_text(page_text)):
                documents.append({
                    "id": f"live-arch-page-chunk-{idx}",
                    "text": chunk,
                    "source": "oceans-niagara.github.io/tools/index.html"
                })
    except Exception as e:
        print(f"Warning: Could not scrape live page: {e}")

    zip_url = "https://github.com/Oceans-Niagara/oceans-niagara.github.io/releases/download/v1.0.0/fleet-mesh.zip"
    print(f"Downloading release zip from {zip_url}...")
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
    else:
        print(f"Failed to download zip. Status code: {response.status_code}")

    print(f"Total chunks across all sources: {len(documents)}. Generating embeddings with rate limits...")

    vectors_to_upsert = []
    for idx, doc in enumerate(documents):
        # Throttle requests to prevent hitting the 100 requests/minute free tier cap
        if idx > 0 and idx % 15 == 0:
            time.sleep(10)
        else:
            time.sleep(0.5)

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=doc["text"]
            config={"output_dimensionality": 768}
        )
        
        if hasattr(response, 'embeddings') and response.embeddings:
            embedding_values = response.embeddings[0].values
        else:
            embedding_values = response.embedding.values

        vectors_to_upsert.append({
            "id": doc["id"],
            "values": embedding_values,
            "metadata": {"text": doc["text"], "source": doc["source"]}
        })

    for i in range(0, len(vectors_to_upsert), 50):
        batch = vectors_to_upsert[i:i+50]
        index.upsert(vectors=batch)
    
    print("Successfully synced live site page and release scripts to Pinecone!")

if __name__ == "__main__":
    process_and_upload()
