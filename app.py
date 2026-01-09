import streamlit as st
import pickle
from core_logic import parse_markdown_to_tree, tree_search, display_tree

st.set_page_config(page_title="Tree-RAG Explorer", layout="wide")

st.title("🌳 Tree-Based RAG (No Vector DB)")
st.markdown("Upload a Markdown file to build a hierarchical search tree.")

# --- Sample Data ---
SAMPLE_MD = """
# Space Station Alpha: Operations Manual
Welcome to the primary manual for the Alpha station.

## Life Support Systems
Systems responsible for maintaining a habitable environment.
### Oxygen Generation
The VOGS (Vortex Oxygen Generation System) uses electrolysis to split water into hydrogen and oxygen.
### Temperature Control
Maintained via liquid ammonia heat exchangers located in the outer hull.

## Power Grid
The station is powered by solar arrays and backup fuel cells.
### Solar Array Alpha
Consists of 48 photovoltaic panels. It must be oriented within 15 degrees of the sun.
### Emergency Fuel Cells
Hydrogen-oxygen fuel cells that activate if the station enters the Earth's shadow for more than 45 minutes.

## Emergency Protocols
Procedures for critical system failures.
### Depressurization
In case of hull breach, seal all magnetic hatches and move to Sector 4.
### Fire Suppression
Uses nitrogen displacement. Do not enter the area without an oxygen mask.
"""
# what should be the angle of the cells?
# --- Updated Sidebar ---
with st.sidebar:
    st.header("Data Source")
    
    # Add a button for Sample Data
    if st.button("Load Sample Manual"):
        with st.spinner("Building tree from sample..."):
            st.session_state.tree = parse_markdown_to_tree(SAMPLE_MD)
        st.success("Sample Tree Loaded!")

    uploaded_file = st.file_uploader("Or Upload Your Own Markdown", type=["md"])
    if uploaded_file:
        md_text = uploaded_file.read().decode("utf-8")
        with st.spinner("Building semantic tree..."):
            st.session_state.tree = parse_markdown_to_tree(md_text)
        st.success("Tree Built!")

if 'tree' not in st.session_state:
    st.session_state.tree = None

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Markdown", type=["md"])
    if uploaded_file:
        md_text = uploaded_file.read().decode("utf-8")
        with st.spinner("Building semantic tree..."):
            st.session_state.tree = parse_markdown_to_tree(md_text)
        st.success("Tree Built!")

# Chat Interface
if st.session_state.tree:
    query = st.text_input("Ask a question about your document:")
    
    if query:
        with st.spinner("Searching tree levels..."):
            result_node = tree_search(st.session_state.tree, query)
            
        st.subheader(f"Relevant Section: {result_node.title}")
        st.info(f"**Path Level:** {result_node.level}")
        st.markdown(result_node.content)
        
    with st.expander("View Technical Metadata"):
        tab1, tab2 = st.tabs(["Search Path 🛣️", "Full Document Tree 🌳"])
        
        with tab1:
            st.write("### How the algorithm navigated your query:")
            # Display the path taken during the last search
            if 'search_history' in st.session_state:
                for step in st.session_state.search_history:
                    st.write(f"**Level {step['level']}**: Selected *{step['winner']}*")
                    # Show top 3 candidates for transparency
                    st.table(step['candidates'][:3]) 
        
        with tab2:
            st.write("### Complete Document Hierarchy:")
            display_tree(st.session_state.tree)
else:
    st.info("Please upload a Markdown file in the sidebar to begin.")