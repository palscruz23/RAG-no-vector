import streamlit as st
import pickle
from dotenv import load_dotenv
from core_logic import parse_markdown_to_tree, tree_search, display_tree, convert_to_markdown_with_progress, convert_to_markdown_timed, parse_markdown_to_tree_timed, get_embedding_model

# Load environment variables from .env file
load_dotenv()

# Use this everywhere in your app
model = get_embedding_model()

if 'tree' not in st.session_state:
    st.session_state.tree = None

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

if 'raw_md' not in st.session_state:
    st.session_state.raw_md = None

if 'times' not in st.session_state:
    st.session_state.times = {"ocr": 0, "tree": 0}

if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None

st.set_page_config(page_title="Tree-RAG Explorer", layout="wide")

st.title("🌳 Tree-Based RAG (No Vector DB)")
st.markdown("Upload a Markdown file to build a hierarchical search tree.")

# --- Sample Data ---
SAMPLE_MD_PATH = "knowledge base/manual_for_pumps.md"

# --- Updated Sidebar ---
with st.sidebar:
    st.header("Data Source")

    # Add a button for Sample Data
    if st.button("Load Sample Manual"):
        with st.spinner("Building tree from sample..."):
            # Load markdown from file
            with open(SAMPLE_MD_PATH, 'r', encoding='utf-8') as f:
                sample_md_text = f.read()
            st.session_state.tree = parse_markdown_to_tree(sample_md_text)
            st.session_state.raw_md = sample_md_text
        st.success("Sample Tree Loaded!")

    # uploaded_file = st.file_uploader("Upload Markdown", type=["md"])
    # if uploaded_file:
    #     md_text = uploaded_file.read().decode("utf-8")
    #     with st.spinner("Building semantic tree..."):
    #         st.session_state.tree = parse_markdown_to_tree(md_text)
    #     st.success("Tree Built!")
        
    with st.status("🚀 Initializing RAG Pipeline...", expanded=True) as status_model:
        # Ensure model is ready (will be instant if already cached)
        st.write("🧠 Checking embedding model...")
        model = get_embedding_model()
        status_model.update(label=f"✨ Model loaded", state="complete", expanded=False)

    # Change the type to ["pdf"]
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

    if uploaded_file:
        # Check if this is a new file (not already processed)
        file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else uploaded_file.name

        if st.session_state.processed_file != file_id:
            my_progress_bar = st.progress(0, text="Preparing document...")

            with st.status("🚀 Converting PDF with Mistral OCR...", expanded=True) as conversion_status:
                md_text, ocr_time = convert_to_markdown_timed(uploaded_file, my_progress_bar, status=conversion_status)
                st.session_state.times['ocr'] = ocr_time
                st.session_state.raw_md = md_text
                conversion_status.update(label=f"✅ Conversion complete ({ocr_time:.2f}s)", state="complete")

            with st.spinner("Building semantic tree...", show_time=True):
                tree, tree_time = parse_markdown_to_tree_timed(md_text)
                st.session_state.times['tree'] = tree_time
                st.session_state.tree = tree

            # Mark this file as processed
            st.session_state.processed_file = file_id

            # Show completion message
            total_time = ocr_time + tree_time
            st.success(f"✨ Processing Complete! (Conversion: {ocr_time:.2f}s, Tree: {tree_time:.2f}s, Total: {total_time:.2f}s)")
        else:
            st.success("✅ Document already processed!")

        # Preview Window (show regardless of whether we just processed or not)
        if st.session_state.raw_md:
            with st.expander("View Raw Markdown Preview 📄"):
                st.code(st.session_state.raw_md[:2000], language="markdown")

# Chat Interface
if st.session_state.tree:
    query = st.text_input("Ask a question about your document:")
    
    if query:
        with st.spinner("Searching tree levels..."):
            result_node = tree_search(st.session_state.tree, query, model=model)
            
        st.subheader(f"Relevant Section: {result_node.title}")
        st.info(f"**Path Level:** {result_node.level}")
        st.markdown(result_node.content)
        
    with st.expander("View Technical Metadata"):
        tab1, tab2 = st.tabs(["Search Path 🛣️", "Document Hierarchy 🌳"])

        with tab1:
            st.write("### How the algorithm navigated your query:")
            # Display the path taken during the last search
            if 'search_history' in st.session_state and st.session_state.search_history:
                for step in st.session_state.search_history:
                    st.write(f"**Level {step['level']}**: Selected *{step['winner']}*")
                    # Show top 3 candidates for transparency
                    st.table(step['candidates'][:3])
            else:
                st.info("Submit a query to see the search path visualization")

        with tab2:
            st.write("### Complete Document Tree:")
            if query:
                display_tree(st.session_state.tree, query=query, model=model)
            else:
                display_tree(st.session_state.tree)
else:
    st.info("Please upload a Markdown file in the sidebar to begin.")