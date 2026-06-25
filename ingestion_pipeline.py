import os
from langchain_community.document_loaders import TextLoader ,  DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()



def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    #check if the docs_path exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist. Please create it and add your company files.")
    
    #Load all .txt files from the docs directory
    loader = DirectoryLoader(
        docs_path, 
        glob="*.txt", 
        loader_cls=TextLoader
        )
    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    



    for i, docs in enumerate(documents[:2]):
        print(f"\n Document {i+1}:")
        print(f"  Source: {docs.metadata['source']}")
        print(f"  Content length: {len(docs.page_content)} characters")
        print(f"  Content preview: {docs.page_content[:100]}...")
        print(f"  Metadata: {docs.metadata}")

    return documents


def split_documents(documents, chunk_size=200, chunk_overlap=0):
    """Split documents into chunks with overlap."""
    print(f"Splitting documents into chunks ...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n Chunk {i+1}:")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"  length: {len(chunk.page_content)} characters")
            print("  Content:")
            print(chunk.page_content)
            print("_" * 50)
        if len(chunks) > 5:
            print(f"\n... {len(chunks) - 5} more chunks")

    return chunks


def main():
    print("Main function")
    #1. Loading the files
    documents = load_documents(docs_path="docs")
    #2. Chunking the files
    chunks = split_documents(documents)
    print(f"Total number of chunks: {len(chunks)}")
    
    #3. Embedding and storing in vector DB
    if len(chunks) > 0:
        print("\nCreating embeddings and vector database...")
        
        # Check if API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "your-" in api_key or "api-key-here" in api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file. Please add your actual API key.")
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Create and persist vector database
        persist_directory = "./chroma_db"
        print(f"Creating vector database at {persist_directory}...")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        print(f"✓ Vector database created successfully!")
        print(f"  - Total vectors stored: {len(chunks)}")
        print(f"  - Location: {persist_directory}")
    else:
        print("No chunks created. Vector database not created.")


if __name__ == "__main__":
    main()