import os
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from decouple import config

# Flag to toggle between local and remote parsing
USE_LOCAL_PARSER = False

# Directory to save downloaded audio files
SAVE_DIRECTORY = "youtube/videos"

# Set OpenAI API key from environment variables
os.environ["API_KEY"] = config("API_KEY")

# YouTube video URL(s)
YOUTUBE_URLS = ["https://www.youtube.com/watch?v=BFezdkKq7LI"]

# Vector store settings
VECTOR_STORE_DIRECTORY = "vector_store_0003"
COLLECTION_NAME = "youtube_video"

try:
    # Initialize the loader with either local or remote Whisper parsing
    if USE_LOCAL_PARSER:
        loader = GenericLoader(
            YoutubeAudioLoader(YOUTUBE_URLS, SAVE_DIRECTORY),
            OpenAIWhisperParserLocal()
        )
    else:
        loader = GenericLoader(
            YoutubeAudioLoader(YOUTUBE_URLS, SAVE_DIRECTORY),
            OpenAIWhisperParser()
        )
    
    # Load and transcribe YouTube audio to text
    documents = loader.load()

    # Combine all documents into a single string
    combined_text = " ".join(doc.page_content for doc in documents)

    # Split the combined text into chunks
    TEXT_CHUNK_SIZE = 1500
    TEXT_CHUNK_OVERLAP = 150
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP
    )
    text_chunks = text_splitter.split_text(combined_text)

    # Initialize embedding function
    embedding_model_name = "all-MiniLM-L6-v2"
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Create and persist a vector database
    vectordb = Chroma.from_texts(
        text_chunks,
        embedding_function,
        persist_directory=VECTOR_STORE_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    print(f"Vector database created successfully in: {VECTOR_STORE_DIRECTORY}")

except Exception as error:
    # Handle exceptions with detailed output
    print(f"An error occurred: {error}")

