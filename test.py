import re
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def is_youtube_link(text):
    """Check if the input text is a valid YouTube link"""
    youtube_pattern = (
        r'(https?://)?(www\.|m\.)?'  # Added 'm\.' for mobile URLs
        r'(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)'
        r'([a-zA-Z0-9_-]+)'
    )
    logging.debug(f"Matching regex against: {text}")
    match = re.match(youtube_pattern, text)
    logging.debug(f"Match result: {match}")
    return bool(match)

if __name__ == '__main__':
    test_links = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "http://youtube.com/watch?v=dQw4w9WgXcQ",
        "youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
        "https://youtu.be/8k8ZT01whqw",
        "https://www.youtube.com/watch?v=jNQXAC9IVRwV5",
         "https://youtu.be/jNQXAC9IVRwV5",
        "https://www.youtube.com/watch?v=shortID",
        "https://youtu.be/shortID",
          "https://www.youtube.com/watch?v=jNQXAC9IVRwV5&ab_channel=testchannel", #With query parameters
         "https://www.youtube.com/watch?v=jNQXAC9IVRwV5?t=10", #With query parameters
        "invalid url",
        "https://www.notyoutube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/8k8ZT01whqw"
    ]

    for link in test_links:
        print(f"{link}: {is_youtube_link(link)}")