import os
import json
from datetime import datetime

import streamlit as st
import pymupdf
import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

GEMINI_MODEL = "gemini-3.5-flash"
HF_MODEL = "openai/gpt-oss-20b"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "resume_knowledge_base"

TOP_K_CHUNKS = 10
TOP_K_QA_CHUNKS = 5
TOP_CANDIDATES = 3


# ============================================================
# CANDIDATE MAPPING
# ============================================================

CANDIDATE_NAMES = {
    "candidate_01": "Arjun Sharma",
    "candidate_02": "Priya Nair",
    "candidate_03": "Rahul Mehta",
    "candidate_04": "Vikram Rao",
    "candidate_05": "Sneha Kapoor",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-Powered Resume Matching",
    layout="wide"
)


# ============================================================
# LOAD GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini_client():

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini configuration unavailable."
        )

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# LOAD HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def load_huggingface_client():

    if not HF_TOKEN:
        raise RuntimeError(
            "Fallback configuration unavailable."
        )

    return InferenceClient(
        api_key=HF_TOKEN
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_chroma_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# INITIALIZE RESOURCES
# ============================================================

embedding_model = load_embedding_model()
collection = load_chroma_collection()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.read()

    doc = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page in doc:
        pages.append(
            page.get_text()
        )

    doc.close()

    return "\n".join(pages)


# ============================================================
# PHASE 1 — JD RETRIEVAL
# ============================================================

def retrieve_resume_chunks(
    jd_text,
    top_k=TOP_K_CHUNKS
):

    jd_embedding = embedding_model.encode(
        jd_text
    )

    results = collection.query(
        query_embeddings=[
            jd_embedding.tolist()
        ],
        n_results=top_k
    )

    retrieved_chunks = []

    for i in range(
        len(results["documents"][0])
    ):

        metadata = results["metadatas"][0][i]

        retrieved_chunks.append(
            {
                "rank": i + 1,
                "candidate": metadata["candidate"],
                "section": metadata["section"],
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i]
            }
        )

    return retrieved_chunks


# ============================================================
# PHASE 2 — CANDIDATE-SPECIFIC RETRIEVAL
# ============================================================

def retrieve_candidate_chunks(
    question,
    candidate_id,
    top_k=TOP_K_QA_CHUNKS
):

    candidate_name = CANDIDATE_NAMES.get(
        candidate_id,
        candidate_id
    )

    search_text = (
        f"{candidate_name} {question}"
    )

    question_embedding = embedding_model.encode(
        search_text
    )

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=top_k,
        where={
            "candidate": candidate_id
        }
    )

    retrieved_chunks = []

    if not results["documents"]:
        return retrieved_chunks

    for i in range(
        len(results["documents"][0])
    ):

        metadata = results["metadatas"][0][i]

        retrieved_chunks.append(
            {
                "candidate": candidate_id,
                "section": metadata["section"],
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i]
            }
        )

    return retrieved_chunks


# ============================================================
# DETECT CANDIDATES IN QUESTION
# ============================================================

def detect_candidates_in_question(
    question,
    top_candidates
):

    question_lower = question.lower()

    detected = []

    for candidate in top_candidates:

        candidate_id = candidate[
            "candidate_id"
        ]

        candidate_name = candidate[
            "candidate_name"
        ]

        if candidate_name.lower() in question_lower:

            detected.append(
                candidate_id
            )

    for candidate_id, candidate_name in CANDIDATE_NAMES.items():

        if candidate_name.lower() in question_lower:

            if candidate_id not in detected:

                detected.append(
                    candidate_id
                )

    return detected


# ============================================================
# GROUP EVIDENCE BY CANDIDATE
# ============================================================

def group_evidence_by_candidate(
    retrieved_chunks
):

    candidate_evidence = {}

    for chunk in retrieved_chunks:

        candidate_id = chunk["candidate"]

        candidate_name = CANDIDATE_NAMES.get(
            candidate_id,
            candidate_id
        )

        if candidate_id not in candidate_evidence:

            candidate_evidence[candidate_id] = {
                "name": candidate_name,
                "evidence": []
            }

        candidate_evidence[
            candidate_id
        ]["evidence"].append(
            chunk
        )

    return candidate_evidence


# ============================================================
# FORMAT CANDIDATE EVIDENCE
# ============================================================

def format_candidate_evidence(
    candidate_evidence
):

    formatted = []

    for candidate_id, data in candidate_evidence.items():

        evidence_text = []

        for item in data["evidence"]:

            evidence_text.append(
                f"""
Section: {item["section"]}

Evidence:
{item["document"]}
"""
            )

        formatted.append(
            f"""
Candidate ID: {candidate_id}
Candidate Name: {data["name"]}

Resume Evidence:
{"".join(evidence_text)}
"""
        )

    return "\n".join(formatted)


# ============================================================
# FORMAT TOP CANDIDATES
# ============================================================

def format_top_candidates(
    top_candidates
):

    return json.dumps(
        top_candidates,
        indent=2
    )


# ============================================================
# GEMINI — TOP 3
# ============================================================

def generate_top_candidates_with_gemini(
    jd_text,
    candidate_evidence
):

    print(
        f"[TOP 3] Trying Gemini: {GEMINI_MODEL}"
    )

    client = load_gemini_client()

    evidence_text = format_candidate_evidence(
        candidate_evidence
    )

    prompt = f"""
You are an AI recruiting assistant.

Analyze the Job Description and the retrieved
resume evidence.

Identify the 3 most relevant candidates.

Base your answer ONLY on the supplied evidence.

Do not invent qualifications.

Job Description:
{jd_text}

Candidate Evidence:
{evidence_text}

Return exactly 3 candidates when at least
3 candidates are available.

Return concise recruiter-friendly reasons.

Return JSON only.
"""

    candidate_schema = {
        "type": "OBJECT",
        "properties": {
            "candidates": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "candidate_id": {
                            "type": "STRING"
                        },
                        "candidate_name": {
                            "type": "STRING"
                        },
                        "reason": {
                            "type": "STRING"
                        }
                    },
                    "required": [
                        "candidate_id",
                        "candidate_name",
                        "reason"
                    ]
                }
            }
        },
        "required": [
            "candidates"
        ]
    }

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=candidate_schema
        )
    )

    result = json.loads(
        response.text
    )

    print(
        "[TOP 3] Gemini succeeded."
    )

    return result["candidates"][:TOP_CANDIDATES]


# ============================================================
# HUGGING FACE — TOP 3 FALLBACK
# ============================================================

def generate_top_candidates_with_huggingface(
    jd_text,
    candidate_evidence
):

    print(
        f"[TOP 3] Trying Hugging Face fallback: "
        f"{HF_MODEL}"
    )

    client = load_huggingface_client()

    evidence_text = format_candidate_evidence(
        candidate_evidence
    )

    prompt = f"""
You are an AI recruiting assistant.

Analyze the Job Description and retrieved
resume evidence.

Identify the 3 most relevant candidates.

Base your answer ONLY on the supplied evidence.

Do not invent qualifications.

Job Description:
{jd_text}

Candidate Evidence:
{evidence_text}

Return exactly this JSON structure:

{{
    "candidates": [
        {{
            "candidate_id": "candidate_xx",
            "candidate_name": "Candidate Name",
            "reason": "Concise recruiter-friendly reason"
        }}
    ]
}}
"""

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a recruiter-support AI. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1500
    )

    result = json.loads(
        response.choices[0].message.content
    )

    print(
        "[TOP 3] Hugging Face fallback succeeded."
    )

    return result["candidates"][:TOP_CANDIDATES]


# ============================================================
# TOP 3 AUTOMATIC FALLBACK
# ============================================================

def generate_top_candidates(
    jd_text,
    candidate_evidence
):

    try:

        return generate_top_candidates_with_gemini(
            jd_text,
            candidate_evidence
        )

    except Exception as gemini_error:

        print(
            f"[TOP 3] Gemini failed: "
            f"{gemini_error}"
        )

    try:

        return generate_top_candidates_with_huggingface(
            jd_text,
            candidate_evidence
        )

    except Exception as hf_error:

        print(
            f"[TOP 3] Hugging Face fallback failed: "
            f"{hf_error}"
        )

    print(
        "[TOP 3] Both models failed."
    )

    raise RuntimeError(
        "Both AI models failed."
    )


# ============================================================
# PHASE 2 — GEMINI Q&A
# ============================================================

def generate_followup_with_gemini(
    jd_text,
    candidate_evidence,
    top_candidates,
    conversation
):

    print(
        f"[Q&A] Trying Gemini: {GEMINI_MODEL}"
    )

    client = load_gemini_client()

    evidence_text = format_candidate_evidence(
        candidate_evidence
    )

    top_candidate_text = format_top_candidates(
        top_candidates
    )

    conversation_text = ""

    for message in conversation:

        conversation_text += (
            f'{message["role"].upper()}: '
            f'{message["content"]}\n\n'
        )

    prompt = f"""
You are an AI recruiting assistant helping a recruiter
evaluate candidates for a Job Description.

Answer the recruiter's latest question using ONLY
the supplied information.

Available information:

1. Job Description
2. Candidate resume evidence
3. Top candidate analysis
4. Conversation history

Do not invent candidate information.

If the requested information is not available,
clearly say that it is not available.

Be concise and recruiter-friendly.

Job Description:
{jd_text}

Candidate Evidence:
{evidence_text}

Top Candidates:
{top_candidate_text}

Conversation History:
{conversation_text}

Answer the latest recruiter question.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    print(
        "[Q&A] Gemini succeeded."
    )

    return response.text.strip()


# ============================================================
# PHASE 2 — HUGGING FACE Q&A
# ============================================================

def generate_followup_with_huggingface(
    jd_text,
    candidate_evidence,
    top_candidates,
    conversation
):

    print(
        f"[Q&A] Trying Hugging Face fallback: "
        f"{HF_MODEL}"
    )

    client = load_huggingface_client()

    evidence_text = format_candidate_evidence(
        candidate_evidence
    )

    top_candidate_text = format_top_candidates(
        top_candidates
    )

    conversation_text = ""

    for message in conversation:

        conversation_text += (
            f'{message["role"].upper()}: '
            f'{message["content"]}\n\n'
        )

    prompt = f"""
You are an AI recruiting assistant.

Answer the recruiter's latest question using ONLY
the supplied Job Description, candidate evidence,
top candidate analysis, and conversation history.

Do not invent candidate information.

If information is unavailable, say so clearly.

Job Description:
{jd_text}

Candidate Evidence:
{evidence_text}

Top Candidates:
{top_candidate_text}

Conversation History:
{conversation_text}

Answer concisely and professionally.
"""

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a recruiter-support AI. "
                    "Answer only using supplied information."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    print(
        "[Q&A] Hugging Face fallback succeeded."
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Q&A AUTOMATIC FALLBACK
# ============================================================

def generate_followup_answer(
    jd_text,
    candidate_evidence,
    top_candidates,
    conversation
):

    try:

        return generate_followup_with_gemini(
            jd_text,
            candidate_evidence,
            top_candidates,
            conversation
        )

    except Exception as gemini_error:

        print(
            f"[Q&A] Gemini failed: "
            f"{gemini_error}"
        )

    try:

        return generate_followup_with_huggingface(
            jd_text,
            candidate_evidence,
            top_candidates,
            conversation
        )

    except Exception as hf_error:

        print(
            f"[Q&A] Hugging Face fallback failed: "
            f"{hf_error}"
        )

    print(
        "[Q&A] Both models failed."
    )

    raise RuntimeError(
        "Both AI models failed."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "chats" not in st.session_state:

    st.session_state.chats = []


if "active_chat_id" not in st.session_state:

    st.session_state.active_chat_id = None


# ============================================================
# CREATE NEW CHAT
# ============================================================

def create_new_chat():

    chat_id = (
        f"chat_"
        f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    )

    new_chat = {
        "id": chat_id,
        "title": "New JD",
        "created_at": datetime.now(),

        "jd_filename": None,
        "jd_text": "",

        "retrieved_chunks": [],
        "candidate_evidence": {},

        "top_candidates": [],

        "messages": []
    }

    st.session_state.chats.append(
        new_chat
    )

    st.session_state.active_chat_id = (
        chat_id
    )


# ============================================================
# GET ACTIVE CHAT
# ============================================================

def get_active_chat():

    for chat in st.session_state.chats:

        if (
            chat["id"]
            == st.session_state.active_chat_id
        ):

            return chat

    return None


# ============================================================
# INITIAL CHAT
# ============================================================

if not st.session_state.chats:

    create_new_chat()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Chat History")

    if st.button(
        "+ New JD",
        use_container_width=True
    ):

        create_new_chat()

        st.rerun()

    st.divider()

    for chat in reversed(
        st.session_state.chats
    ):

        title = chat["title"]

        if st.button(
            title,
            key=f'chat_{chat["id"]}',
            use_container_width=True
        ):

            st.session_state.active_chat_id = (
                chat["id"]
            )

            st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "AI-Powered Resume Matching"
)


chat = get_active_chat()


if chat is None:

    create_new_chat()

    chat = get_active_chat()


# ============================================================
# JD UPLOAD
# ============================================================

if not chat["top_candidates"]:

    st.subheader(
        "Job Description"
    )

    uploaded_file = st.file_uploader(
        "Upload JD PDF",
        type=["pdf"],
        key=f'upload_{chat["id"]}'
    )

    if uploaded_file:

        chat["jd_filename"] = (
            uploaded_file.name
        )

        if chat["title"] == "New JD":

            chat["title"] = (
                uploaded_file.name
                .replace(".pdf", "")
            )

        if st.button(
            "Find relevant resumes",
            type="primary",
            use_container_width=True
        ):

            try:

                # --------------------------------------------
                # Extract JD
                # --------------------------------------------

                jd_text = extract_text_from_pdf(
                    uploaded_file
                )

                if not jd_text.strip():

                    st.error(
                        "The uploaded JD could not be processed."
                    )

                    st.stop()

                chat["jd_text"] = jd_text


                # --------------------------------------------
                # User-friendly processing indicator
                # --------------------------------------------

                with st.spinner(
                    "Finding relevant candidates..."
                ):

                    # ----------------------------------------
                    # Retrieve JD-relevant chunks
                    # ----------------------------------------

                    print(
                        "\n=============================="
                    )

                    print(
                        "[RAG] Starting JD retrieval..."
                    )

                    retrieved_chunks = (
                        retrieve_resume_chunks(
                            jd_text,
                            TOP_K_CHUNKS
                        )
                    )

                    chat["retrieved_chunks"] = (
                        retrieved_chunks
                    )


                    # ----------------------------------------
                    # Group evidence
                    # ----------------------------------------

                    candidate_evidence = (
                        group_evidence_by_candidate(
                            retrieved_chunks
                        )
                    )

                    chat["candidate_evidence"] = (
                        candidate_evidence
                    )


                    # ----------------------------------------
                    # Generate Top 3
                    # ----------------------------------------

                    top_candidates = (
                        generate_top_candidates(
                            jd_text,
                            candidate_evidence
                        )
                    )

                    chat["top_candidates"] = (
                        top_candidates
                    )


                    # ----------------------------------------
                    # Initial message
                    # ----------------------------------------

                    chat["messages"] = [
                        {
                            "role": "assistant",
                            "content": (
                                "Top 3 candidates generated."
                            )
                        }
                    ]


                    print(
                        "[RAG] Top 3 analysis complete."
                    )

                    print(
                        "==============================\n"
                    )


                st.rerun()


            except Exception:

                print(
                    "[ERROR] Resume matching failed."
                )

                st.error(
                    "We couldn't complete the analysis "
                    "right now. Please try again in a moment."
                )


# ============================================================
# DISPLAY TOP 3
# ============================================================

if chat["top_candidates"]:

    st.subheader(
        "Top 3 Relevant Candidates"
    )

    for index, candidate in enumerate(
        chat["top_candidates"],
        start=1
    ):

        st.markdown(
            f"### {index}. "
            f"{candidate['candidate_name']}"
        )

        st.write(
            candidate["reason"]
        )


    # ========================================================
    # FOLLOW-UP CHAT
    # ========================================================

    st.divider()


    # --------------------------------------------------------
    # DISPLAY PREVIOUS CONVERSATION
    # --------------------------------------------------------

    for message in chat["messages"]:

        if message["content"] == (
            "Top 3 candidates generated."
        ):
            continue

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    user_question = st.chat_input(
        "Ask about the candidates..."
    )


    if user_question:

        # ----------------------------------------------------
        # Display the user message IMMEDIATELY
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.write(
                user_question
            )


        # ----------------------------------------------------
        # Candidate detection
        # ----------------------------------------------------

        detected_candidates = (
            detect_candidates_in_question(
                user_question,
                chat["top_candidates"]
            )
        )


        if detected_candidates:

            print(
                "\n=============================="
            )

            print(
                f"[Q&A] Question: {user_question}"
            )

            print(
                "[Q&A] Candidate-specific question."
            )

            print(
                "[Q&A] Candidates detected: "
                + ", ".join(
                    CANDIDATE_NAMES.get(
                        candidate_id,
                        candidate_id
                    )
                    for candidate_id
                    in detected_candidates
                )
            )

        else:

            print(
                "\n=============================="
            )

            print(
                f"[Q&A] Question: {user_question}"
            )

            print(
                "[Q&A] No specific candidate detected."
            )


        # ----------------------------------------------------
        # Retrieve additional evidence
        # ----------------------------------------------------

        qa_evidence = {}


        if detected_candidates:

            for candidate_id in detected_candidates:

                print(
                    f"[Q&A RAG] Searching "
                    f"{CANDIDATE_NAMES.get(candidate_id, candidate_id)}..."
                )

                chunks = retrieve_candidate_chunks(
                    user_question,
                    candidate_id,
                    TOP_K_QA_CHUNKS
                )

                qa_evidence[
                    candidate_id
                ] = {
                    "name": CANDIDATE_NAMES.get(
                        candidate_id,
                        candidate_id
                    ),
                    "evidence": chunks
                }

                print(
                    f"[Q&A RAG] Retrieved "
                    f"{len(chunks)} chunks."
                )

        else:

            qa_evidence = (
                chat["candidate_evidence"]
            )


        # ----------------------------------------------------
        # Save question BEFORE generating response
        # ----------------------------------------------------

        chat["messages"].append(
            {
                "role": "user",
                "content": user_question
            }
        )


        # ----------------------------------------------------
        # Generate AI response
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Loading AI response..."
            ):

                try:

                    answer = generate_followup_answer(
                        chat["jd_text"],
                        qa_evidence,
                        chat["top_candidates"],
                        chat["messages"]
                    )


                    chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                    st.write(
                        answer
                    )


                    print(
                        "[Q&A] Response generated successfully."
                    )

                    print(
                        "==============================\n"
                    )


                except Exception:

                    # Remove question if response failed
                    chat["messages"].pop()

                    print(
                        "[Q&A] Unable to generate response."
                    )

                    print(
                        "==============================\n"
                    )

                    st.error(
                        "We couldn't complete the analysis "
                        "right now. Please try again in a moment."
                    )

