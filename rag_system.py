import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fuzzywuzzy import fuzz


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Generate embeddings for text chunks
embedding_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

def generate_embeddings(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Create FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Search FAISS index
def search_faiss_index(index, query_embedding, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

# Query FAISS and return results
def process_query(index, query, chunks):
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = search_faiss_index(index, np.array(query_embedding))
    results = [chunks[i] for i in indices[0]]
    return results

def should_query_faiss(query: str) -> bool:
    simple_queries = ["hello", "hi", "how are you", "thank you", "bye"]
    threshold = 80  # Define a similarity threshold
    for simple_query in simple_queries:
        if fuzz.ratio(query.lower(), simple_query.lower()) > threshold:
            return False
    return True

def calculator(query: str) -> str:
    """Evaluates mathematical expressions in the query."""
    try:
        result = eval(query, {"__builtins__": None}, {})
        return f"The result of the calculation is {result}."
    except Exception as e:
        return f"Unable to process the calculation: {e}"
    

import requests

def serpapi_web_search(query: str, api_key: str = '60c5cb0e2c062bb81764465d95a6189ffcbd6c12443e37879c7a7d486a094400', num_results: int = 3) -> str:
    """Perform a web search using SerpAPI and return the top results."""
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "hl": "en",  # Language
        "gl": "in",  # Country (e.g., 'in' for India)
        "num": num_results,
        "api_key": api_key,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "organic_results" in data:
            results = [
                f"- {result['title']} ({result['link']})"
                for result in data["organic_results"][:num_results]
            ]
            return "\n".join(results) if results else "No relevant results found."
        return "No relevant results found."
    except requests.exceptions.RequestException as e:
        return f"Error during web search: {str(e)}"


import requests
import base64

def text_to_speech(text: str, api_key: str, target_language="en-IN", speaker="meera"):
    """Convert text to speech using Sarvam's TTS API."""
    url = "https://api.sarvam.ai/text-to-speech"
    missing_padding = len(text) % 4
    if missing_padding:
        text += ' ' * (4 - missing_padding)
    text = base64.b64decode(text)
    
    payload = {
        "inputs": [text],
        "target_language_code": target_language,
        "speaker": speaker,
        "pitch": 0,  # Adjust as needed
        "pace": 1.0,  # Adjust as needed
        "loudness": 1.0,  # Adjust as needed
        "speech_sample_rate": 22050,  # You can adjust this based on your requirements
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key}
    
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        # Parse the base64 audio response
        audio_data = response.json().get('audios', [None])[0]
        if audio_data:
            return audio_data  # Base64 encoded audio file
        else:
            return "Error: No audio data received."
    else:
        return f"Error: {response.status_code}, {response.text}"




def dynamic_agent(query: str, index, chunks) -> str:
    """Uses LLM to decide which agents to invoke dynamically."""
    
    # Step 1: Retrieve document-relevant context using FAISS
    document_context = ""
    if should_query_faiss(query):
        relevant_chunks = process_query(index, query, chunks)
        document_context = "\n".join(relevant_chunks)
    
    system_prompt = (
    "You are an intelligent assistant. Based on the user's query, determine if you need to:"
    "\n1. Use the provided document context (if relevant) to answer directly. Provide a clear, detailed, and relevant answer from the context."
    "\n2. Perform a calculation. In this case, include <CALCULATOR:expression> in your response."
    "\n3. Perform a web search for external knowledge. In this case, include <WEBSEARCH:query> in your response."
    "\nIf external knowledge or context is required, your response should include relevant details for the user's query."
    "\nYour response should be detailed and informative but **avoid any unnecessary reasoning** or decision-making explanation."
    "\nDo not give extra text regarding your thought process; only return the relevant placeholders or a direct, informative answer."
    "\nEnsure the response provides the user with enough context, facts, or explanation for a complete answer."
)



    # Step 3: Construct the User Prompt
    user_prompt = (
        f"Document Context: {document_context}\n"  # Include document context if found relevant
        f"Query: {query}\n"
        "Please decide if you need to use the document context, a calculator, or perform a web search."
        "Respond with the necessary placeholders if tools are required."
    )
    
    # Step 4: Get Response from the Language Model (LLM)
    from groq import Groq
    client = Groq(api_key="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")
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model="llama3-70b-8192",  # Example LLM model
        temperature=0.5,
        max_tokens=500,
        top_p=1,
    ).choices[0].message.content

    # Step 5: Check if Response Requires Calculator or Web Search
    # Handle the case where a calculator is needed
    if "<CALCULATOR:" in response:
        expression = response.split("<CALCULATOR:")[1].split(">")[0].strip()
        calc_result = calculator(expression)
        response = response.replace(f"<CALCULATOR:{expression}>", calc_result)
    
    

    # Handle Web Search Placeholder
    if "<WEBSEARCH:" in response:
        try:
            search_query = response.split("<WEBSEARCH:")[1].split(">")[0].strip()
            web_result = serpapi_web_search(search_query)
            response = response.replace(f"<WEBSEARCH:{search_query}>", web_result)
        except Exception as e:
            response += f"\n[Error in Web Search: {str(e)}]"
    
    return response



import json
import re
import streamlit as st

def generate_quiz_from_chunks(chunks, num_questions=5, max_chunk_size=22000):
    """
    Generate quiz questions from the provided text chunks by summarizing the content.

    Args:
        chunks (list): List of text chunks from the uploaded PDFs.
        num_questions (int): Number of quiz questions to generate.
        max_chunk_size (int): Maximum token size for input text.

    Returns:
        list: List of quiz questions, each containing a question, options, and the correct answer.
    """
    from groq import Groq

    # Initialize the Groq client
    client = Groq(api_key="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")

    # Step 1: Summarize chunks to fit within token limits
    summarized_text = ""
    for chunk in chunks:
        if len(summarized_text) + len(chunk) > max_chunk_size:
            break  # Limit size of summarized input
        summarized_text += f" {chunk}"

    # Step 2: Construct the quiz generation prompt
    system_prompt = (
        "You are an educational assistant. Generate quiz questions with multiple-choice answers."
    )
    user_prompt = f"""
    Based on the following text, generate {num_questions} quiz questions. Each question must include:
    1. The question text.
    2. Four multiple-choice options.
    3. The correct option clearly indicated.

    Return the quiz in the following JSON format:
    {{
        "questions": [
            {{
                "question": "Question text here",
                "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                "correct_answer": "Option 1"
            }},
            ... (more questions)
        ]
    }}

    Text: {summarized_text}
    """

    try:
        # Step 3: Generate the quiz using the LLM
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=500,
            top_p=1,
        ).choices[0].message.content

        
        # Step 4: Extract JSON using regex
        match = re.search(r"\{.*\}", response, re.DOTALL)  # Extract JSON object
        if not match:
            raise ValueError("No valid JSON object found in the LLM response.")
        
        quiz_json = match.group(0)  # Extract matched JSON

        # Parse JSON safely
        try:
            quiz_data = json.loads(quiz_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        # Validate JSON structure
        if not isinstance(quiz_data, dict) or "questions" not in quiz_data:
            raise ValueError("Parsed JSON does not contain a valid 'questions' field.")

        questions = quiz_data["questions"]
        if not isinstance(questions, list):
            raise ValueError("The 'questions' field must be a list.")

        # Return validated questions
        return questions

    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        raise RuntimeError(f"Quiz generation failed: {e}")
