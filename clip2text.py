import os
import moviepy as mp
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence


# 1. Extract audio from the mp4 video
def extract_audio_from_video(video_path, audio_path):
    # Load video file
    video = mp.VideoFileClip(video_path)
    # Extract audio from video
    audio = video.audio
    # Write audio to file
    audio.write_audiofile(audio_path)

# 2. Convert audio to text (transcription)
def transcribe_audio(audio_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Load audio file using SpeechRecognition
    audio_file = sr.AudioFile(audio_path)
    
    with audio_file as source:
        audio_data = recognizer.record(source)
    
    # Try to recognize the speech in the audio
    # try:
    transcript = recognizer.recognize_google(audio_data)
    return transcript
    # except sr.UnknownValueError:
    #     return "Sorry, I couldn't understand the audio."
    # except sr.RequestError:
    #     return "Sorry, the service is unavailable."

# 3. Store the transcript in a text file
def save_transcript_to_file(transcript, file_path):
    with open(file_path, 'w') as file:
        file.write(transcript)
    print(f"Transcript saved to {file_path}")

# Main function to integrate the steps
def process_video(video_path, audio_path, transcript_file_path):
    # Extract audio from video
    extract_audio_from_video(video_path, audio_path)
    print(f"Audio extracted and saved to {audio_path}")
    
    # Transcribe the audio
    transcript = get_large_audio_transcription_on_silence(audio_path)
    print("Audio transcription complete.")
    
    # Save the transcript to a text file
    save_transcript_to_file(transcript, transcript_file_path)


def get_large_audio_transcription_on_silence(path):
    """Splitting the large audio file into chunks and apply speech recognition on each of these chunks"""
    # Open the audio file using pydub
    sound = AudioSegment.from_file(path)  
    
    # Split audio sound where silence is 500 milliseconds or more and get chunks
    chunks = split_on_silence(
        sound,
        min_silence_len=500,
        silence_thresh=sound.dBFS-14,
        keep_silence=500,
    )
    
    folder_name = "audio-chunks"
    
    # Create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    whole_text = ""
    
    # Process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # Export audio chunk and save it in the `folder_name` directory
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        
        # Recognize the chunk
        try:
            text = transcribe_audio(chunk_filename)
        except sr.UnknownValueError as e:
            print(f"Error with {chunk_filename}: {str(e)}")
            text = ""
        else:
            text = f"{text.capitalize()}. "
            print(chunk_filename, ":", text)
            whole_text += text
        
        # Delete the chunk after processing
        os.remove(chunk_filename)
        print(f"Deleted {chunk_filename} after processing.")
    
    # Return the full text for all chunks detected
    return whole_text



if __name__ == "__main__":
    # Example usage
    video_path = "youtube/videos/How to Speak So That People Want to Listen ｜ Julian Treasure ｜ TED.mp4"
    audio_path = "youtube/dubs/extracted_audio.wav"
    transcript_file_path = "youtube/dubs/transcript.txt"
    process_video(video_path, audio_path, transcript_file_path)
    get_large_audio_transcription_on_silence(audio_path)
