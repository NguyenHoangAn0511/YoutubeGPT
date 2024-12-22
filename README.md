# YouTubeGPT: Video/Audio Question Answering Chatbot

This project implements a question-answering chatbot that can extract information from YouTube videos and local audio/video files. It uses a Retrieval-Augmented Generation (RAG) approach with a large language model for answering questions, and utilizes Streamlit for the user interface.

**Note**: This project was created as part of an assignment. To adhere to best practices, I have chosen to load the Google API Key from a `.env` file for Conda and environment variables for Docker. Please ensure you set up your environment as described below.

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Setup with Conda](#setup-with-conda)
3.  [Using a .env File for API Key (Conda)](#using-a-env-file-for-api-key-conda)
4.  [Setup with Docker](#setup-with-docker)
5.  [Running the Application](#running-the-application)
6.  [How to Use](#how-to-use)
7.  [Troubleshooting](#troubleshooting)

## 1. Prerequisites

Before you begin, make sure you have the following installed:

*   **Conda:** If you plan to use Conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
*   **Docker:** If you plan to use Docker, install [Docker Desktop](https://www.docker.com/products/docker-desktop) or [Docker Engine](https://docs.docker.com/engine/install/).
*   **Google API Key:** You will need a Google API Key to use the Gemini API. You can get one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## 2. Setup with Conda

Follow these steps to create a Conda environment with Python 3.13.3 and install all the project dependencies:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/NguyenHoangAn0511/YoutubeGPT.git
    cd YoutubeGPT
    ```

2.  **Create the Conda Environment:**

    ```bash
    conda create -n youtubegpt python=3.13.3
    conda activate youtubegpt
    ```
    This command creates a new conda environment named `youtubegpt` with python 3.13.3 and activates it.

3. **Install project requirements:**
     ```bash
    pip install -r requirements.txt
    ```

## 3. Using a .env File for API Key (Conda)

Instead of setting an environment variable directly in your shell, this project loads the `GOOGLE_API_KEY` from a `.env` file.

1.  **Create a `.env` file**: Create a file named `.env` in the root directory of your project (the same directory where your `app.py` and `utils.py` files are located).

2.  **Add your API key:** Add the following line to your `.env` file, replacing `<your_api_key>` with your actual Google API key:

    ```
    GOOGLE_API_KEY="<your_api_key>"
    ```

    It is important that there are no spaces before and after the equal symbol, and that it is placed inside double quotes.

    *Note: Ensure that you add the `.env` file to your `.gitignore` to avoid committing your API key to your code repository.*

## 4. Setup with Docker

Follow these steps to build and run the application using Docker:

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_project_directory>
    ```

    Replace `<your_repository_url>` with the URL of your project's repository and `<your_project_directory>` with the directory name.

2.  **Set the API Key:**

    Set the `GOOGLE_API_KEY` environment variable in your shell:

    ```bash
    export GOOGLE_API_KEY="<your_api_key>"
    ```
     Replace `<your_api_key>` with the Google API key you obtained earlier.
3.  **Build the Docker Image:**

    From the root directory of your project (the same directory where your `Dockerfile` and `docker-compose.yml` files are located), run the following command:

    ```bash
    docker-compose build
    ```
     This builds the Docker image using the instructions in your `Dockerfile`.
4.  **Run the Docker Container:**

    After building, start the Docker container using:

    ```bash
    docker-compose up
    ```
    This command will start the container and your app.

    The app will be accessible on `http://localhost:8501`.

## 5. Running the Application

### Running with Conda

To start the Streamlit application using conda, ensure the conda environment `youtubegpt` is activated, then run:

```bash
streamlit run app.py