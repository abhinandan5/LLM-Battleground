import streamlit as st
import json
import os
import sys
import tempfile
import time
import shutil

# --- SYS-PATH ROUTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- IMPORTS ---
from src.engine.retriever import process_document
from src.engine.models import get_gpt_response, get_gemini_response
from src.eval.judge import evaluate_responses
from src.utils.helpers import extract_json
from src.eval.validation import validate_output
from src.utils.exporter import export_chat_history
# MCP disabled for cloud deployment
# from src.engine.mcp_client import get_context_from_mcp
# from src.engine.ingest import ingest_data 

# --- PAGE CONFIG & CUSTOM CSS (BEAUTIFICATION) ---
st.set_page_config(page_title="RAG Arena", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #303046;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .gpt-header { color: #4A90E2; font-weight: bold; }
    .gemini-header { color: #F5A623; font-weight: bold; }
    .streamlit-expanderHeader { font-weight: bold; color: #A0A0B5; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "scoreboard" not in st.session_state:
    st.session_state.scoreboard = {"GPT": {"acc": [], "comp": []}, "Gemini": {"acc": [], "comp": []}}

st.title("LLM Arena (GPT Vs Gemini)")
st.markdown("Upload documents and evaluate multi-agent orchestration pipelines dynamically.")

# --- SIDEBAR & SCOREBOARD ---
with st.sidebar:
    st.header("⚙️ Backend Engine")
    st.info("Using In-Memory Vector Database")
    backend_engine = "Original In-Memory DB"
    
    st.divider()
    st.header("1. Data Ingestion")
    
    tab1, tab2 = st.tabs(["Local File", "YouTube URL"])
    with tab1:
        file_input = st.file_uploader("Upload Context", type=["pdf", "txt", "csv", "png", "jpg", "jpeg"])
    with tab2:
        url_input = st.text_input("Paste YouTube Video Link")
        
    uploaded_file = file_input
    youtube_url = url_input if url_input and url_input.strip() != "" else None
    
    # --- NEW: DATABASE MANAGEMENT ---
    st.divider()
    st.header("🧹 Database Management")
    
    if st.button("🗑️ Clear Database", use_container_width=True):
        st.session_state.retriever = None
        st.session_state.messages = []
        st.session_state.current_file = None
        st.success("Database cleared!")

    st.divider()
    st.header("2. Pipeline Settings")
    selected_judge = st.selectbox(
        "Select Evaluator:",
        options=["Claude 4.6 Sonnet (Best)", "Claude 3 Haiku", "GPT-4o-mini (OpenAI)"]
    )
    
    st.divider()
    st.header("Live Scoreboard")
    
    def calc_avg(score_list):
        return sum(score_list) / len(score_list) if score_list else 0.0

    gpt_avg_acc, gpt_avg_comp = calc_avg(st.session_state.scoreboard["GPT"]["acc"]), calc_avg(st.session_state.scoreboard["GPT"]["comp"])
    gem_avg_acc, gem_avg_comp = calc_avg(st.session_state.scoreboard["Gemini"]["acc"]), calc_avg(st.session_state.scoreboard["Gemini"]["comp"])

    sc1, sc2 = st.columns(2)
    sc1.metric("GPT Acc", f"{gpt_avg_acc:.1f}")
    sc1.metric("GPT Comp", f"{gpt_avg_comp:.1f}")
    sc2.metric("Gemini Acc", f"{gem_avg_acc:.1f}")
    sc2.metric("Gemini Comp", f"{gem_avg_comp:.1f}")
    
    st.divider()
    st.header("3. Audit & Export")
    csv_data = export_chat_history(st.session_state.messages)
    if csv_data:
        st.download_button(
            label="Download Arena Report (CSV)",
            data=csv_data,
            file_name="rag_evaluation_audit.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- DOCUMENT PROCESSING ---
active_source = None
is_url = False

if uploaded_file:
    active_source = uploaded_file.name
elif youtube_url:
    active_source = youtube_url
    is_url = True

if active_source is not None:
    if st.session_state.current_file != active_source:
        st.session_state.current_file = active_source
        st.session_state.messages = [] 
        st.session_state.scoreboard = {"GPT": {"acc": [], "comp": []}, "Gemini": {"acc": [], "comp": []}}
        
        # --- DOCUMENT INGESTION ---
        with st.spinner("Building VectorDB Pipeline..."):
            if is_url:
                st.session_state.retriever = process_document(youtube_url)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{active_source.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                st.session_state.retriever = process_document(tmp_file_path)
                os.unlink(tmp_file_path) 
        st.success(f"✅ Ingested: {active_source}")

# --- UI RENDER HELPER ---
def render_arena_turn(gpt_ans, gemini_ans, eval_data, gpt_valid, gemini_valid):
    st.success(f"**Verdict:** {eval_data.get('Winner', 'Tie')}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 class='gpt-header'> GPT Output</h3>", unsafe_allow_html=True)
        if not gpt_valid[0]: st.warning(f" Validation Flag: {gpt_valid[1]}")
        st.write(gpt_ans)
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{eval_data.get('Model A', {}).get('Accuracy', '-')} / 10")
        m2.metric("Completeness", f"{eval_data.get('Model A', {}).get('Completeness', '-')} / 10")
        st.info(f"**Feedback:** {eval_data.get('Model A', {}).get('Feedback', '')}")
    
    with col2:
        st.markdown("<h3 class='gemini-header'> Gemini Output</h3>", unsafe_allow_html=True)
        if not gemini_valid[0]: st.warning(f" Validation Flag: {gemini_valid[1]}")
        st.write(gemini_ans)
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{eval_data.get('Model B', {}).get('Accuracy', '-')} / 10")
        m2.metric("Completeness", f"{eval_data.get('Model B', {}).get('Completeness', '-')} / 10")
        st.info(f"**Feedback:** {eval_data.get('Model B', {}).get('Feedback', '')}")

# --- CHAT INTERFACE ---
if st.session_state.retriever is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                with st.expander(" Pipeline Trace: Retrieved Context"):
                    for i, chunk in enumerate(msg["chunks"]):
                        st.code(f"Context Node {i+1}\n{chunk}", language="text")
                render_arena_turn(msg["gpt_ans"], msg["gemini_ans"], msg["eval_data"], msg["gpt_valid"], msg["gemini_valid"])
            
    if question := st.chat_input("Test the pipeline..."):
        with st.chat_message("user"):
            st.markdown(question)
            
        st.session_state.messages.append({"role": "user", "content": question})
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[:-1]])

        with st.chat_message("assistant"):
            
            with st.spinner("Retrieving context..."):
                if st.session_state.retriever is None:
                    st.error("Please upload a document or URL first.")
                    st.stop()
                docs = st.session_state.retriever.invoke(question)
                chunk_texts = [d.page_content for d in docs]
                context = "\n\n".join([f"Chunk {i+1}:\n{text}" for i, text in enumerate(chunk_texts)]) 
                
            with st.spinner("Generating Agent Responses..."):
                gpt_ans = get_gpt_response(context, question, history_str)
                gemini_ans = get_gemini_response(context, question, history_str)
                
                gpt_valid = validate_output(gpt_ans)
                gemini_valid = validate_output(gemini_ans)
                
            with st.spinner(f"Evaluating via {selected_judge}..."):
                evaluation = None
                for attempt in range(3):
                    try:
                        evaluation = evaluate_responses(context, question, gpt_ans, gemini_ans, judge_choice=selected_judge)
                        break 
                    except Exception as e:
                        if attempt < 2: time.sleep(2)
                        else: st.error(f"Judge API failed: {e}")

            if evaluation:
                eval_data = extract_json(evaluation)
                if eval_data:
                    try:
                        st.session_state.scoreboard["GPT"]["acc"].append(float(eval_data.get("Model A", {}).get("Accuracy", 0)))
                        st.session_state.scoreboard["GPT"]["comp"].append(float(eval_data.get("Model A", {}).get("Completeness", 0)))
                        st.session_state.scoreboard["Gemini"]["acc"].append(float(eval_data.get("Model B", {}).get("Accuracy", 0)))
                        st.session_state.scoreboard["Gemini"]["comp"].append(float(eval_data.get("Model B", {}).get("Completeness", 0)))
                    except: pass 
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"GPT: {gpt_ans}\nGemini: {gemini_ans}",
                        "question": question, 
                        "gpt_ans": gpt_ans,
                        "gemini_ans": gemini_ans,
                        "eval_data": eval_data,
                        "chunks": chunk_texts,
                        "gpt_valid": gpt_valid, 
                        "gemini_valid": gemini_valid
                    })
                    st.rerun()
                else:
                    st.error("Evaluation Parse Error.")