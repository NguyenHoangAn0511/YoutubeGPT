import shutil
import yt_dlp
import os
import re
import streamlit as st
from decouple import config
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from clip2text import get_large_audio_transcription_on_silence, process_video
from chromadb import Settings
import logging
import chromadb
import uuid
import tempfile


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def is_youtube_link(text):
    """Check if the input text is a valid YouTube link"""
    youtube_pattern = (
        r'(https?://)?(www\.|m\.)?'
        r'(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)'
        r'([a-zA-Z0-9_-]+)'
    )
    logging.debug(f"Matching regex against: {text}")
    match = re.match(youtube_pattern, text)
    logging.debug(f"Match result: {match}")
    return bool(match)


def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.

    Args:
        url (str): The YouTube video URL.

    Returns:
        str: The YouTube video ID, or None if no valid ID is found.
    """
    # Regular expression to match video ID after 'v=' in the YouTube URL
    match = re.search(r'[?&]v=([a-zA-Z0-9_-]+)', url)
    
    if match:
        return match.group(1)
    else:
        print("Invalid YouTube URL or video ID not found.")
        return None


def save_youtube_transcript(video_id, output_file):
    """
    Fetches the transcript of a YouTube video, formats it, and saves it to a file.

    Args:
        video_id (str): The YouTube video ID.
        output_file (str): The path to the output file where the formatted transcript will be saved.

    Returns:
        str: The formatted transcript as a single string.
    """
    try:
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format transcript
        formatter = TextFormatter()
        formatted_transcript = formatter.format_transcript(transcript).replace('\n', ' ')
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(formatted_transcript)
        
        print(f"Transcript successfully saved to '{output_file}'.")
        return formatted_transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def download_video_with_cookies(youtube_url, save_directory="youtube/videos", cookies_file="cookies.txt"):
    """
    Downloads a video from YouTube using yt-dlp with cookie authentication.

    Parameters:
    - youtube_url (str): The URL of the YouTube video to download.
    - save_directory (str): Directory where the video will be saved. Defaults to "downloaded_videos".
    - cookies_file (str): Path to the cookies.txt file. Defaults to "cookies.txt".

    Returns:
    - str: The path to the downloaded video file.
    """
    try:
        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Set yt-dlp options with cookies file
        ydl_options = {
            "format": "best[ext=mp4]/worst[ext=mp4]",  # Select lowest quality MP4 to save storage
            "outtmpl": os.path.join(save_directory, "%(title)s.%(ext)s"),  # Save path and file name
            "cookiefile": cookies_file,  # Use the exported cookies.txt file
        }

        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            print(f"Video downloaded successfully: {downloaded_file}")
            return downloaded_file

    except Exception as error:
        print(f"Failed to download the video: {error}")
        return None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def set_google_api_key():
    """
    Set the Google API key from the environment using configuration.
    """
    os.environ['GOOGLE_API_KEY'] = config("GOOGLE_API_KEY")


def download_and_process_video(video_url):
    """
    Downloads the video and processes its transcript.
    
    Args:
        video_url (str): The URL of the YouTube video.
    
    Returns:
        str: The video ID extracted from the URL.
    """
    # Download the video (Assumed function exists)
    video_path = download_video_with_cookies(video_url)
    
    # Extract video ID
    video_id = extract_video_id(video_url)
    
    try:
        # Save transcript to file
        save_youtube_transcript(video_id, f"youtube/dubs/{video_id}.txt")
        print(f"Transcript successfully saved for video ID: {video_id}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the transcript: {e}")
        return None, video_path
    return f"youtube/dubs/{video_id}.txt", video_path


def load_transcript(transcript_path):
    """
    Loads the transcript from the saved file.

    Args:
        video_id (str): The ID of the YouTube video.

    Returns:
        list: A list of documents containing the transcript.
    """
    loader = TextLoader(transcript_path)
    return loader.load()


def initialize_embeddings():
    """
    Initializes the embedding model.

    Returns:
        GoogleGenerativeAIEmbeddings: The embeddings model.
    """
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def create_vectorstore(docs, gemini_embeddings, collection_name="default_collection"):
    """
    Creates or updates a vector store for document retrieval.

    Args:
        docs (list): A list of documents to be indexed.
        gemini_embeddings (GoogleGenerativeAIEmbeddings): The embeddings model.
        collection_name (str): The name of the collection to use.
    
    Returns:
        Chroma: The persisted vector store.
    """
    client = chromadb.Client()

    # Create Chroma vectorstore
    return Chroma.from_documents(
        client = client,
        documents=docs, 
        embedding=gemini_embeddings,
        collection_name = collection_name,
    )


def setup_retriever(vectorstore):
    """
    Loads the persisted vector store and sets up the retriever.

    Returns:
        Chroma: The vector store with the retriever.
    """
    return vectorstore.as_retriever(search_kwargs={"k": 1})


def create_llm():
    """
    Creates and returns the LLM (Language Model) used for question answering.

    Returns:
        ChatGoogleGenerativeAI: The chat model for Q&A.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        top_p=0.85
    )


def create_prompt_template():
    """
    Creates and returns the prompt template used by the LLM.

    Returns:
        PromptTemplate: The prompt template.
    """
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use ten sentences maximum and keep the answer concise.\n
    Question: {question} \nContext: {context} \nAnswer:"""
    
    return PromptTemplate.from_template(llm_prompt_template)


def create_rag_chain(retriever, llm_prompt, llm, conversation_context, user_question):
    """
    Creates a RAG (Retrieval-Augmented Generation) chain for question-answering.

    Args:
        retriever (Chroma): The vector store retriever.
        llm_prompt (PromptTemplate): The LLM prompt template.
        llm (ChatGoogleGenerativeAI): The language model.
        conversation_context (str): The ongoing conversation context.
        user_question (str): The current user question.
    
    Returns:
        LLMChain: The RAG chain to process questions and generate answers.
    """
    # Combine the context with the current conversation
    context_with_conversation = f"{llm_prompt} \n{conversation_context} \nUser: {user_question}"

    # Pass the combined context and question to the LLM
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )


def start_chat_loop_streamlit(retriever, llm_prompt, llm, session_state_key="chat_messages"):
    """
    Initiates a continuous chat loop with a RAG model, integrated with Streamlit.

    Args:
        retriever (Chroma): The vector store retriever.
        llm_prompt (PromptTemplate): The LLM prompt template.
        llm (ChatGoogleGenerativeAI): The language model.
        session_state_key (str, optional): Key to store chat messages in Streamlit's session state. Defaults to "chat_messages".
    """

    # Initialize chat messages in session state if not already present
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = []

    # Display past messages
    for message in st.session_state[session_state_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state[session_state_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot response
        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                rag_chain = (
                    {"context": retriever | format_docs, "question": lambda x: x }
                    | llm_prompt
                    | llm
                    | StrOutputParser()
                 )
                response = rag_chain.invoke(prompt)

            st.session_state[session_state_key].append({"role": "assistant", "content": response})
            st.markdown(response)