import streamlit as st
from rag_system import (
    extract_text_from_pdf,
    chunk_text,
    generate_embeddings,
    create_faiss_index,
    process_query,
    should_query_faiss,
    dynamic_agent,
    text_to_speech,
    generate_quiz_from_chunks
)
import hashlib
import base64
import io

# Utility function to get file hash
def get_file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []  # Stores conversation history
if "chunks" not in st.session_state:
    st.session_state.chunks = []  # Stores document chunks
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None  # Stores FAISS index
if "query_inputs" not in st.session_state:
    st.session_state.query_inputs = []  # Handles multiple queries
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # Tracks processed files

# Streamlit UI
st.title("RAG SYSTEM WITH LLAMA3")
st.subheader("Upload PDFs and Ask Questions")



# File uploader
uploaded_files = st.file_uploader(
    "Drag and drop your PDF files here", accept_multiple_files=True, type=["pdf"]
)

# Process uploaded PDFs and create FAISS index
if uploaded_files:
    st.write("Uploaded Files:")
    all_chunks = []  # Collect all chunks from PDFs
    for file in uploaded_files:
        st.write(f"📄 {file.name}")
        file_hash = get_file_hash(file)

        # Avoid reprocessing files
        if file_hash not in st.session_state.processed_files:
            try:
                pdf_text = extract_text_from_pdf(file)
                chunks = chunk_text(pdf_text)
                all_chunks.extend(chunks)
                st.session_state.processed_files.add(file_hash)

            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

    # Add unique chunks to session state
    st.session_state.chunks.extend([c for c in all_chunks if c not in st.session_state.chunks])

    # Generate embeddings and create FAISS index if not already done
    if not st.session_state.faiss_index and all_chunks:
        st.write("Generating embeddings and creating FAISS index...")
        embeddings = generate_embeddings(st.session_state.chunks)
        st.session_state.faiss_index = create_faiss_index(embeddings)

# Initialize session state variables
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "quiz" not in st.session_state:
    st.session_state.quiz = None

if uploaded_files:
    if st.button("Generate Quiz"):
        if st.session_state.chunks:
            with st.spinner("Generating quiz..."):
                try:
                    quiz_questions = generate_quiz_from_chunks(st.session_state.chunks)
                    st.session_state.quiz = quiz_questions  # Save quiz questions to session state
                    st.success("Quiz generated successfully!")
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")
                    st.info("Please review your input or try again.")
        else:
            st.warning("Please upload and process PDFs first.")

    # Display the quiz questions if they exist
    if st.session_state.quiz:
        st.write("### Quiz Questions:")
        for idx, question in enumerate(st.session_state.quiz, 1):
            with st.expander(f"Question {idx}"):
                st.markdown(f"**{question['question']}**")  # Use markdown for better formatting
                for opt_idx, option in enumerate(question["options"], 1):
                    st.write(f"{opt_idx}. {option}")
                st.markdown(f"**Correct Answer:** {question['correct_answer']}")  # Display the correct answer
    else:
        if "quiz" in st.session_state and not st.session_state.quiz:
            st.info("No quiz has been generated yet. Please click 'Generate Quiz' to create one.")


# Allow user to ask questions
if st.session_state.faiss_index:
    st.write("You can now start asking questions!")

    # Display all previous questions and responses
    for i, question in enumerate(st.session_state.query_inputs):
        st.write(f"**Question {i + 1}:** {question}")
        st.expander(f"Response {i + 1}").write(st.session_state.history[i * 2 + 1]["content"])

        if st.button(f"🔊 Listen to Response {i + 1}", key=f"listen_{i}"):
            with st.spinner("Generating audio..."):
                try:
                    response_text = st.session_state.history[i * 2 + 1]["content"]
                    audio_base64 = text_to_speech(response_text, api_key="2d7fc1dd-d1d3-46cd-b69b-1b8483f356f7")

                    # Decode and stream audio
                    audio_data = base64.b64decode(audio_base64)
                    audio_file = io.BytesIO(audio_data)
                    st.audio(audio_file, format="audio/wav")
                except Exception as e:
                    st.error(f"Error generating audio: {e}")
        
        # Feedback Section
        feedback_key = f"feedback_{i}"  # Unique key for each feedback button
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None  # Initialize feedback state

        feedback = st.radio(
            f"Was the response for Question {i + 1} helpful?", 
            ("Yes", "No"), 
            key=feedback_key
        )

        if feedback:
            if feedback == "No":
                detailed_feedback = st.text_input(
                    f"Tell us how we can improve Response {i + 1}:",
                    key=f"detailed_feedback_{i}"
                )
            else:
                st.write(f"Thank you for your feedback on Response {i + 1}!")


    # Input box for the next question
    query = st.text_input(f"Enter Question {len(st.session_state.query_inputs) + 1}:")
    if st.button(f"Submit Question {len(st.session_state.query_inputs) + 1}"):
        if query.strip():
            st.session_state.query_inputs.append(query)
            response = dynamic_agent(query, st.session_state.faiss_index, st.session_state.chunks)
            st.session_state.history.append({"role": "user", "content": query})
            st.session_state.history.append({"role": "assistant", "content": response})
            st.rerun()
        else:
            st.warning("Please enter a valid question.")
