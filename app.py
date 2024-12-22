import streamlit as st
import chromadb
from utils import (
    set_google_api_key,
    download_and_process_video,
    load_transcript,
    initialize_embeddings,
    create_vectorstore,
    setup_retriever,
    create_llm,
    create_prompt_template,
    is_youtube_link,
    start_chat_loop_streamlit,
    process_video
)
from decouple import config
import os

def main():
    st.title("Video/Audio Question Answering Chatbot")

    # Set up Google API Key
    os.environ['GOOGLE_API_KEY'] = config("GOOGLE_API_KEY")
    
     # Initialize video processing flag if not already present
    if "video_processed" not in st.session_state:
        st.session_state.video_processed = False


    # Initialize chat history if not already present
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Initialize collection name if not already present
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = "default_collection"



    # Button to start a new chat
    if st.button("New Chat"):
        st.session_state.chat_messages = []
        st.session_state.video_processed = False # This will allow for video to be re-processed
        st.session_state.video_url = ""  # Reset video URL
        if 'retriever' in st.session_state:
            del st.session_state.retriever
        if 'llm_prompt' in st.session_state:
            del st.session_state.llm_prompt
        if 'llm' in st.session_state:
            del st.session_state.llm
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        

    # Get the video URL from the user
    if "video_url" not in st.session_state:
        st.session_state.video_url = ""
    video_url = st.text_input("Enter YouTube URL or local file path (audio/video):", value=st.session_state.video_url)
    st.session_state.video_url = video_url
   
    if video_url:
        # Validate YouTube link
        if not is_youtube_link(video_url):
            st.warning("Using local video/audio file.")
        else:
            st.info("Using YouTube video.")
            
        # Process video only once
        if not st.session_state.video_processed:
            with st.spinner('Processing video/audio and setting up the chatbot... Please wait! This may take a while!'):
                # Chat function from utils, start chat
                audio_path = f"youtube/audio/{video_url.split('/')[-1]}.wav"
                if is_youtube_link(video_url):
                    transcript_path, video_path = download_and_process_video(video_url)
                    if transcript_path is None:
                        process_video(video_path, audio_path, transcript_path)
                else:
                    transcript_path = f"youtube/dubs/{video_url.split('/')[-1]}.txt"
                    process_video(video_url, audio_path, transcript_path)

                # Step 3: Load the transcript
                docs = load_transcript(transcript_path)

                # Step 3: Initialize the embeddings model
                gemini_embeddings = initialize_embeddings()
                
                # Step 4: Create and persist the vector store
                vectorstore = create_vectorstore(docs, gemini_embeddings, collection_name= st.session_state.collection_name)
                
                # Step 5: Set up the retriever for document retrieval
                retriever = setup_retriever(vectorstore)
                
                # Step 6: Set up the LLM model
                llm = create_llm()
                
                # Step 7: Create the prompt template
                llm_prompt = create_prompt_template()
                st.session_state.retriever = retriever
                st.session_state.llm_prompt = llm_prompt
                st.session_state.llm = llm
            st.session_state.video_processed = True

    if st.session_state.video_processed:
         start_chat_loop_streamlit(st.session_state.retriever, st.session_state.llm_prompt, st.session_state.llm, session_state_key="chat_messages")

if __name__ == "__main__":
    main()