from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import os

# Loading virtual env
load_dotenv()

# Extracting the saved api key for pinecone
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Loading the data and splitting it into chunks
loader = DirectoryLoader('/Users/adeel/GenAI_Chatbot/University-Course-Books-Chatbot-GENAI', glob="*.pdf",loader_cls=PyPDFLoader)
documents=loader.load() 
text_chunks=text_split(documents)
embeddings = download_hugging_face_embeddings()

# Saving the embeddings in pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "llmcoursebook"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)