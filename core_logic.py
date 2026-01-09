import re
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Load model once to share across nodes
model = SentenceTransformer('all-MiniLM-L6-v2')

class DocNode:
    def __init__(self, title, content, level):
        self.title = title
        self.content = content
        self.level = level
        self.children = []
        self.metadata = {}

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_rrf_score(self, query, k=60):
        if not self.children:
            return []

        # Prepare corpus from search_context (augmented headers)
        corpus = [child.metadata.get('search_context', child.title) for child in self.children]
        
        # 1. BM25 Lexical
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())
        
        # 2. BERT Semantic
        query_enc = model.encode(query)
        child_encs = model.encode(corpus)
        bert_scores = util.cos_sim(query_enc, child_encs)[0].tolist()

        # 3. RRF Fusion
        bm25_rank = np.argsort(bm25_scores)[::-1]
        bert_rank = np.argsort(bert_scores)[::-1]

        rrf_map = {i: 0.0 for i in range(len(self.children))}
        for rank, idx in enumerate(bm25_rank):
            rrf_map[idx] += 1 / (k + rank)
        for rank, idx in enumerate(bert_rank):
            rrf_map[idx] += 1 / (k + rank)

        sorted_indices = sorted(rrf_map, key=rrf_map.get, reverse=True)
        return [(self.children[i], rrf_map[i]) for i in sorted_indices]

def parse_markdown_to_tree(markdown_text):
    header_pattern = re.compile(r'^(#{1,6})\s+(.*)')
    root = DocNode("Root", "Document Start", 0)
    stack = [root]
    lines = markdown_text.split('\n')
    current_content = []

    for line in lines:
        match = header_pattern.match(line)
        if match:
            if current_content:
                stack[-1].content += "\n".join(current_content)
                current_content = []
            level = len(match.group(1))
            new_node = DocNode(match.group(2), "", level)
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            stack[-1].add_child(new_node)
            stack.append(new_node)
        else:
            if line.strip(): current_content.append(line)
    
    if current_content: stack[-1].content += "\n".join(current_content)
    
    # Simple recursive context augmentation (Header + first 100 words)
    def augment(node):
        node.metadata['search_context'] = f"{node.title}: {node.content[:200]}"
        for child in node.children: augment(child)
    augment(root)
    return root

def tree_search(node, query, depth_limit=5):
    # Clear history at the start of a fresh search (at the root)
    if node.level == 0:
        st.session_state.search_history = []

    if not node.children or node.level >= depth_limit:
        return node

    ranked_results = node.get_rrf_score(query)
    if not ranked_results:
        return node

    # Record this step for the visual metadata
    best_child, best_score = ranked_results[0]
    
    # Store candidates as a list of dicts for the table
    candidates = [{"Node": r[0].title, "RRF Score": f"{r[1]:.4f}"} for r in ranked_results]
    
    st.session_state.search_history.append({
        "level": node.level + 1,
        "winner": best_child.title,
        "candidates": candidates
    })

    return tree_search(best_child, query, depth_limit)

def display_tree(node):
    """Recursively displays the document hierarchy using Streamlit expanders."""
    for child in node.children:
        with st.expander(f"L{child.level}: {child.title}"):
            if child.children:
                display_tree(child)
            else:
                st.text(child.content[:200] + "...")

from docling.document_converter import DocumentConverter
import tempfile
import os

# Initialize the converter once
converter = DocumentConverter()

def convert_to_markdown(uploaded_file):
    """Uses Docling to turn a PDF/Image into structured Markdown."""
    # 1. Docling needs a file path, so we save to a temporary file
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # 2. Run the conversion pipeline
        result = converter.convert(tmp_path)
        
        # 3. Export to clean Markdown
        markdown_output = result.document.export_to_markdown()
        return markdown_output
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)