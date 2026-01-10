import streamlit as st
import pickle
from dotenv import load_dotenv
from core_logic import (
    parse_markdown_to_tree, tree_search, tree_search_beam, display_tree,
    convert_to_markdown_with_progress, convert_to_markdown_timed,
    parse_markdown_to_tree_timed, get_embedding_model,
    get_top_k_results, generate_rag_answer, evaluate_with_ragas
)

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

if 'rag_answer' not in st.session_state:
    st.session_state.rag_answer = None

if 'top_k_results' not in st.session_state:
    st.session_state.top_k_results = None

if 'evaluation_scores' not in st.session_state:
    st.session_state.evaluation_scores = None

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

    # Search Configuration
    st.header("Search Settings")

    search_method = st.radio(
        "Search Algorithm",
        options=["Greedy (Fast)", "Beam Search (Thorough)"],
        help="Greedy: Fast single-path search. Beam: Explores multiple paths for better coverage."
    )

    beam_width = 3  # Default value
    if search_method == "Beam Search (Thorough)":
        beam_width = st.slider(
            "Beam Width",
            min_value=2,
            max_value=10,
            value=3,
            help="Number of candidate paths to maintain. Higher = more thorough but slower."
        )

    # RAG Mode Toggle
    use_rag = st.checkbox(
        "Enable RAG Generation",
        value=True,
        help="Generate answers using LLM with top-3 retrieved contexts"
    )

    # Evaluation Toggle
    enable_evaluation = st.checkbox(
        "Enable RAGAS Evaluation",
        value=False,
        help="Evaluate RAG responses with AI judge (slower, uses more API calls)"
    )

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
        if use_rag:
            # RAG Mode: Get top-3 results and generate answer
            with st.spinner("Retrieving top-3 relevant sections..."):
                method = "beam" if search_method == "Beam Search (Thorough)" else "greedy"
                top_k_results = get_top_k_results(
                    st.session_state.tree,
                    query,
                    k=3,
                    method=method,
                    beam_width=beam_width,
                    model=model
                )
                st.session_state.top_k_results = top_k_results

            # Display retrieved sections
            st.subheader("📚 Retrieved Sections (Top 3)")
            for i, (node, score) in enumerate(top_k_results, 1):
                with st.expander(f"{i}. {node.title} (Score: {score:.3f})"):
                    st.markdown(node.content)

            # Generate answer
            with st.spinner("Generating answer with Mistral LLM..."):
                rag_answer = generate_rag_answer(query, top_k_results)
                st.session_state.rag_answer = rag_answer

            # Display generated answer
            st.subheader("💡 Generated Answer using Mistral LLM...")
            st.markdown(rag_answer)

            # Evaluate if enabled
            if enable_evaluation:
                with st.spinner("Evaluating with RAGAS using GPT-4o-mini..."):
                    contexts = [node.content for node, _ in top_k_results]
                    eval_scores = evaluate_with_ragas(query, rag_answer, contexts)
                    st.session_state.evaluation_scores = eval_scores

        else:
            # Original single-result mode
            if search_method == "Beam Search (Thorough)":
                with st.spinner(f"Searching with beam width {beam_width}..."):
                    result_node = tree_search_beam(
                        st.session_state.tree,
                        query,
                        beam_width=beam_width,
                        model=model
                    )
            else:
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
                    # Show beam size if using beam search
                    if 'beam_size' in step:
                        st.write(f"**Level {step['level']}**: Selected *{step['winner']}* (Beam size: {step['beam_size']})")
                    else:
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

    # RAGAS Evaluation Display
    if use_rag and enable_evaluation and st.session_state.evaluation_scores:
        st.divider()
        st.subheader("📊 RAGAS Evaluation Scores using GPT-4o-mini")

        eval_result = st.session_state.evaluation_scores

        # Check if it's an error dict
        if isinstance(eval_result, dict) and "error" in eval_result:
            st.error(f"Evaluation failed: {eval_result['error']}")
        else:
            # Convert RAGAS result to dict
            try:
                # RAGAS returns an EvaluationResult object with a to_pandas() method
                if hasattr(eval_result, 'to_pandas'):
                    scores_df = eval_result.to_pandas()
                    # Get only numeric columns (the actual metric scores)
                    eval_scores = {}
                    for col in scores_df.columns:
                        if col not in ['question', 'answer', 'contexts', 'ground_truth']:
                            try:
                                # Try to convert to numeric
                                score_value = scores_df[col].iloc[0]
                                if isinstance(score_value, (int, float)):
                                    eval_scores[col] = score_value
                            except:
                                pass
                else:
                    eval_scores = eval_result

                # Display scores in columns
                cols = st.columns(len(eval_scores))

                for idx, (metric, score) in enumerate(eval_scores.items()):
                    with cols[idx]:
                        # Handle NaN values
                        import math
                        if math.isnan(score):
                            st.metric(
                                label=metric.replace("_", " ").title(),
                                value="N/A",
                                delta=None
                            )
                            st.write("⚠️ Metric failed to compute")
                        else:
                            # Color code based on score
                            if score >= 0.7:
                                color = "🟢"
                            elif score >= 0.5:
                                color = "🟡"
                            else:
                                color = "🔴"

                            st.metric(
                                label=metric.replace("_", " ").title(),
                                value=f"{score:.3f}",
                                delta=None
                            )
                            st.write(f"{color}")
            except Exception as e:
                st.error(f"Error displaying scores: {str(e)}")

            # Explanation
            with st.expander("📖 Metric Explanations"):
                st.markdown("""
                **Faithfulness**: Does the answer stay grounded in the provided context? (Higher is better)
                - 1.0 = Perfectly grounded, no hallucinations
                - 0.0 = Completely hallucinated

                **Answer Relevancy**: How relevant is the answer to the question? (Higher is better)
                - 1.0 = Directly answers the question
                - 0.0 = Completely irrelevant

                **Context Precision**: How relevant are the retrieved contexts? (Requires ground truth)
                - 1.0 = All retrieved contexts are relevant
                - 0.0 = No relevant contexts retrieved

                **Context Recall**: Did we retrieve all necessary information? (Requires ground truth)
                - 1.0 = All needed information retrieved
                - 0.0 = Missing critical information
                """)

else:
    st.info("Please upload a Markdown file in the sidebar to begin.")