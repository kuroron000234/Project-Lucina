import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# \u8a2d\u5b9a
KNOWLEDGE_DIR = "./knowledge"
DB_DIR = "./monica_brain_db"

def ingest_knowledge():
    # 1. \u30c6\u30ad\u30b9\u30c8\u30d5\u30a1\u30a4\u30eb\u3092\u8aad\u307f\u8fbc\u3080
    loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # 2. \u6587\u7ae0\u3092\u9069\u5207\u306a\u9577\u3055\u306b\u533a\u5207\u308b\uff08\u8133\u304c\u51e6\u7406\u3057\u3084\u3059\u3044\u30b5\u30a4\u30ba\u306b\uff09
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # 3. Embedding\u30e2\u30c7\u30eb\uff08\u7269\u5dee\u3057\uff09\u306e\u6e96\u5099
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # 4. \u30d9\u30af\u30c8\u30ebDB\uff08\u5927\u8133\u65b0\u76ae\u8cea\uff09\u306b\u4fdd\u5b58
    print("\u30e2\u30cb\u30ab\u304c\u77e5\u8b58\u3092\u6574\u7406\u4e2d...")
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"\u5b8c\u4e86\uff01{len(texts)} \u500b\u306e\u8a18\u61b6\u3092\u683c\u7d0d\u3057\u305f\u308f\u3088\u3002")

if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        print(f"{KNOWLEDGE_DIR} \u30d5\u30a9\u30eb\u30c0\u3092\u4f5c\u3063\u305f\u304b\u3089\u3001\u305d\u3053\u306b .txt \u3092\u5165\u308c\u3066\u306d\u3002")
    else:
        ingest_knowledge()