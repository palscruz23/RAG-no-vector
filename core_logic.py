import re
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import tempfile
import os
import time

class DocNode:
    def __init__(self, title, content, level):
        self.title = title
        self.content = content
        self.level = level
        self.children = []
        self.metadata = {}

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_rrf_score(self, query, k=20, model=None, semantic_weight=2.0, lexical_weight=1.0):
        if not self.children:
            return []

        # Get model from parameter or load default
        if model is None:
            model = get_embedding_model()

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

        # 3. Weighted RRF Fusion (semantic weighted 2x higher than lexical)
        bm25_rank = np.argsort(bm25_scores)[::-1]
        bert_rank = np.argsort(bert_scores)[::-1]

        rrf_map = {i: 0.0 for i in range(len(self.children))}
        for rank, idx in enumerate(bm25_rank):
            rrf_map[idx] += lexical_weight * (1 / (k + rank))
        for rank, idx in enumerate(bert_rank):
            rrf_map[idx] += semantic_weight * (1 / (k + rank))

        # 4. Apply softmax to convert scores to probabilities
        scores = np.array([rrf_map[i] for i in range(len(self.children))])

        # Softmax formula: exp(x_i) / sum(exp(x_j))
        # Use numerical stability trick: subtract max before exp
        exp_scores = np.exp(scores - np.max(scores))
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Update rrf_map with softmax probabilities
        rrf_map = {i: softmax_scores[i] for i in range(len(self.children))}

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

def tree_search(node, query, depth_limit=5, model=None):
    # Clear history at the start of a fresh search (at the root)
    if node.level == 0:
        st.session_state.search_history = []

    if not node.children or node.level >= depth_limit:
        return node

    ranked_results = node.get_rrf_score(query, model=model)
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

    return tree_search(best_child, query, depth_limit, model=model)

def tree_search_beam(root, query, beam_width=3, depth_limit=5, model=None):
    """
    Beam search: Maintains multiple candidate paths through the tree.

    Args:
        root: Root node of the tree
        query: Search query string
        beam_width: Number of best candidates to keep at each level (default: 3)
        depth_limit: Maximum depth to search (default: 5)
        model: Embedding model for scoring

    Returns:
        Best leaf node found
    """
    # Clear history at the start
    st.session_state.search_history = []

    # Initialize beam with root and score 0
    beam = [(root, 0.0, [])]  # (node, cumulative_score, path)

    for depth in range(depth_limit):
        candidates = []
        all_children_at_level = []

        # Expand all nodes in current beam
        for node, cum_score, path in beam:
            if not node.children:
                # Leaf node - keep it as a candidate
                candidates.append((node, cum_score, path))
                continue

            # Score children of this node
            ranked_results = node.get_rrf_score(query, model=model)

            if not ranked_results:
                candidates.append((node, cum_score, path))
                continue

            # Add all children as candidates
            for child, score in ranked_results:
                new_path = path + [child.title]
                candidates.append((child, cum_score + score, new_path))
                all_children_at_level.append((child, score))

        if not candidates:
            break

        # Sort by cumulative score and keep top beam_width
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]

        # Record this level for visualization (show all candidates considered)
        if all_children_at_level:
            # Get unique children and their best scores
            child_scores = {}
            for child, score in all_children_at_level:
                if child.title not in child_scores or score > child_scores[child.title]:
                    child_scores[child.title] = score

            # Create candidates list sorted by score
            level_candidates = [{"Node": title, "RRF Score": f"{score:.4f}"}
                              for title, score in sorted(child_scores.items(),
                                                        key=lambda x: x[1],
                                                        reverse=True)]

            # Winner is the top beam candidate at this level
            winner = beam[0][0].title if beam else "None"

            st.session_state.search_history.append({
                "level": depth + 1,
                "winner": winner,
                "candidates": level_candidates,
                "beam_size": len(beam)
            })

        # Check if all beam candidates are leaf nodes
        if all(not node.children for node, _, _ in beam):
            break

    # Return the best node from the final beam
    if beam:
        return beam[0][0]
    else:
        return root

def get_top_k_results(root, query, k=3, method="greedy", beam_width=3, depth_limit=5, model=None):
    """
    Get top-K most relevant document sections for RAG.

    Args:
        root: Root node of the tree
        query: Search query
        k: Number of top results to return (default: 3)
        method: "greedy" or "beam"
        beam_width: Beam width for beam search
        depth_limit: Maximum depth to search
        model: Embedding model

    Returns:
        List of (node, score) tuples, sorted by relevance
    """
    if method == "beam":
        # Use beam search to get diverse candidates
        st.session_state.search_history = []
        beam = [(root, 0.0, [])]

        all_leaf_candidates = []

        for depth in range(depth_limit):
            candidates = []

            for node, cum_score, path in beam:
                if not node.children:
                    # Leaf node - add to final candidates
                    all_leaf_candidates.append((node, cum_score))
                    candidates.append((node, cum_score, path))
                    continue

                ranked_results = node.get_rrf_score(query, model=model)
                if not ranked_results:
                    candidates.append((node, cum_score, path))
                    continue

                for child, score in ranked_results:
                    new_path = path + [child.title]
                    new_cum_score = cum_score + score
                    candidates.append((child, new_cum_score, new_path))

                    # If child is a leaf, add to candidates
                    if not child.children:
                        all_leaf_candidates.append((child, new_cum_score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width * 2]  # Keep more candidates for diversity

            if all(not node.children for node, _, _ in beam):
                break

        # Get top k from all leaf candidates
        all_leaf_candidates.sort(key=lambda x: x[1], reverse=True)
        return all_leaf_candidates[:k]

    else:
        # Greedy search: get best path, then get siblings at each level
        best_path = []
        current_node = root

        # First, get the greedy path
        for depth in range(depth_limit):
            if not current_node.children:
                break

            ranked_results = current_node.get_rrf_score(query, model=model)
            if not ranked_results:
                break

            best_child, best_score = ranked_results[0]
            best_path.append((best_child, best_score))
            current_node = best_child

        # Now collect top candidates from the final level
        if best_path:
            final_node = best_path[-1][0]
            parent = root

            # Navigate to parent of final node
            for node, _ in best_path[:-1]:
                parent = node

            # Get all siblings scored
            if parent.children:
                ranked_results = parent.get_rrf_score(query, model=model)
                return ranked_results[:k]

        # Fallback: just return the single result
        return [(current_node, 1.0)]

def display_tree(node, prefix="", is_last=True, query=None, model=None):
    """Recursively displays the document hierarchy in folder structure format with relevance scores."""

    # Calculate scores for all children if query is provided
    scores_map = {}
    if query and model and node.children:
        try:
            # Get RRF scores for this level
            ranked_results = node.get_rrf_score(query, model=model)
            for child, score in ranked_results:
                scores_map[child.title] = score
        except:
            pass

    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)

        # Determine the tree characters
        if is_last_child:
            connector = "└── "
            extension = "    "
        else:
            connector = "├── "
            extension = "│   "

        # Create the folder/file icon based on whether it has children
        icon = "📁" if child.children else "📄"

        # Add score badge if available with level-specific styling
        score_badge = ""
        if child.title in scores_map:
            score = scores_map[child.title]

            # Different visual indicators based on tree level
            if child.level == 1:
                # Top level - use large indicators
                if score > 0.5:
                    score_badge = f" 🔥 **[{score:.3f}]**"
                elif score > 0.3:
                    score_badge = f" ⭐ **[{score:.3f}]**"
                else:
                    score_badge = f" **[{score:.3f}]**"
            elif child.level == 2:
                # Second level - use medium indicators
                if score > 0.5:
                    score_badge = f" 🟥 *<{score:.3f}>*"
                elif score > 0.3:
                    score_badge = f" 🟨 *<{score:.3f}>*"
                else:
                    score_badge = f" *<{score:.3f}>*"
            elif child.level == 3:
                # Third level - use small indicators
                if score > 0.5:
                    score_badge = f" 🔴 ({score:.3f})"
                elif score > 0.3:
                    score_badge = f" 🟡 ({score:.3f})"
                else:
                    score_badge = f" ({score:.3f})"
            else:
                # Deeper levels - minimal indicators
                if score > 0.5:
                    score_badge = f" • {score:.3f}"
                elif score > 0.3:
                    score_badge = f" ◦ {score:.3f}"
                else:
                    score_badge = f" {score:.3f}"

        # Display the current node
        st.markdown(f"`{prefix}{connector}{icon} {child.title}`{score_badge}")

        # Recursively display children
        if child.children:
            display_tree(child, prefix=prefix + extension, is_last=is_last_child, query=query, model=model)

import streamlit as st
import time

def convert_to_markdown_with_progress(uploaded_file, progress_bar, status=None):
    # Create temporary file for PDF processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Import required libraries
        import fitz  # PyMuPDF
        import base64

        progress_bar.progress(0.1, text="Opening PDF...")

        # Open PDF with PyMuPDF
        pdf_document = fitz.open(tmp_path)
        total_pages = len(pdf_document)

        progress_bar.progress(0.2, text=f"Analyzing {total_pages} pages...")

        all_markdown = []
        ocr_pages = []  # Track which pages need OCR

        # First pass: Try fast text extraction with PyMuPDF
        for page_num in range(total_pages):
            page = pdf_document[page_num]

            # Extract text directly from PDF
            text = page.get_text()

            # Check if page has meaningful text (more than just whitespace/special chars)
            has_text = len(text.strip()) > 50  # Threshold for "enough text"

            if has_text:
                # Page has extractable text - use fast extraction
                all_markdown.append(text)
            else:
                # Page needs OCR - mark it for processing
                ocr_pages.append(page_num)
                all_markdown.append(None)  # Placeholder

            # Update progress
            progress = 0.2 + (0.3 * (page_num + 1) / total_pages)
            progress_bar.progress(progress, text=f"Analyzing page {page_num + 1}/{total_pages}...")

        # Second pass: Use Mistral OCR only for pages that need it
        if ocr_pages:
            from mistralai import Mistral

            # Initialize Mistral client
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment variables. Please set it in .env file.")

            client = Mistral(api_key=api_key)

            progress_bar.progress(0.5, text=f"Using AI OCR for {len(ocr_pages)} complex pages...")

            for idx, page_num in enumerate(ocr_pages):
                page = pdf_document[page_num]

                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")

                # Convert to base64 for Mistral API
                img_base64 = base64.b64encode(img_data).decode('utf-8')

                # Update progress
                progress = 0.5 + (0.4 * (idx + 1) / len(ocr_pages))
                progress_bar.progress(progress, text=f"OCR page {page_num + 1}/{total_pages} (AI)...")

                # Get context from previous page if available
                context = ""
                if page_num > 0 and all_markdown[page_num - 1]:
                    prev_text = all_markdown[page_num - 1]
                    # Get last 500 chars as context
                    context = prev_text[-500:] if len(prev_text) > 500 else prev_text

                # Use Mistral Pixtral for OCR with context
                prompt = "Extract all text from this image and format it as markdown. Preserve the document structure including headings, lists, and formatting. Do NOT add 'Page X' headers - maintain the natural document flow."

                if context:
                    prompt += f"\n\nContext from previous page:\n{context}\n\nEnsure continuity with the previous content."

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{img_base64}"
                            }
                        ]
                    }
                ]

                # Call Mistral API
                chat_response = client.chat.complete(
                    model="pixtral-12b-2409",
                    messages=messages
                )

                page_markdown = chat_response.choices[0].message.content
                all_markdown[page_num] = page_markdown

        pdf_document.close()

        progress_bar.progress(0.9, text="Merging pages...")

        # Merge all pages into a single document
        raw_text = "\n\n".join([md for md in all_markdown if md is not None])

        # Post-process to improve document structure
        import re

        # Remove multiple consecutive blank lines (keep max 2)
        cleaned_text = re.sub(r'\n{4,}', '\n\n', raw_text)

        # Remove page numbers at the start/end of sections if present
        cleaned_text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', cleaned_text)

        # Check if we need to add structure to the document
        # If there are no markdown headers, use AI to structure it
        has_structure = bool(re.search(r'^#{1,6}\s+.+$', cleaned_text, re.MULTILINE))

        print(f"DEBUG: has_structure={has_structure}, text_length={len(cleaned_text)}")

        if not has_structure and len(cleaned_text.strip()) > 100:
            # Use Mistral to add document structure
            # Only for documents under 50k chars to avoid issues
            if len(cleaned_text) <= 50000:
                progress_bar.progress(0.95, text="Adding document structure with AI...")
                if status:
                    status.update(label="🤖 Adding AI structure...", state="running")

                from mistralai import Mistral
                api_key = os.getenv("MISTRAL_API_KEY")

                if api_key:
                    try:
                        client = Mistral(api_key=api_key)

                        # Process in chunks if needed
                        chunk_size = 15000
                        if len(cleaned_text) <= chunk_size:
                            # Process whole document
                            if status:
                                st.write(f"📝 Structuring entire document ({len(cleaned_text)} chars)...")

                            messages = [
                                {
                                    "role": "user",
                                    "content": f"""Convert this plain text document into well-structured markdown with proper headings, sections, and formatting.

Rules:
- Identify main topics and create hierarchical headings (# for title, ## for sections, ### for subsections)
- Preserve all content - don't summarize or omit anything
- Format lists, tables, and code blocks appropriately
- Keep the original text intact, just add markdown structure
- Use proper markdown syntax

Document text:
{cleaned_text}"""
                                }
                            ]

                            chat_response = client.chat.complete(
                                model="mistral-large-latest",
                                messages=messages
                            )

                            cleaned_text = chat_response.choices[0].message.content
                            if status:
                                st.write(f"✅ Document structured!")
                        else:
                            # For larger docs, process in chunks with context
                            chunks = []
                            num_chunks = (len(cleaned_text) + chunk_size - 1) // chunk_size

                            if status:
                                st.write(f"📚 Processing large document in {num_chunks} chunks ({len(cleaned_text)} chars total)...")

                            for i in range(num_chunks):
                                start = i * chunk_size
                                end = min((i + 1) * chunk_size, len(cleaned_text))
                                chunk = cleaned_text[start:end]

                                # Get context from previous chunk if available
                                context = ""
                                if i > 0:
                                    prev_chunk = chunks[-1]
                                    context = f"\n\nPrevious section ended with:\n{prev_chunk[-500:]}"

                                progress = 0.95 + (0.04 * i / num_chunks)
                                progress_text = f"Structuring chunk {i+1}/{num_chunks}..."
                                progress_bar.progress(progress, text=progress_text)

                                if status:
                                    st.write(f"🔄 Chunk {i+1}/{num_chunks}: Processing {len(chunk)} chars...")

                                prompt = f"""Convert this plain text chunk into well-structured markdown. This is part {i+1} of {num_chunks}.

Rules:
- Add hierarchical headings (# for title, ## for sections, ### for subsections)
- Preserve all content - don't summarize
- Continue from previous section naturally{context}

Text chunk:
{chunk}"""

                                messages = [{"role": "user", "content": prompt}]
                                chat_response = client.chat.complete(
                                    model="mistral-large-latest",
                                    messages=messages
                                )

                                chunk_result = chat_response.choices[0].message.content
                                chunks.append(chunk_result)

                                if status:
                                    st.write(f"✅ Chunk {i+1}/{num_chunks} complete ({len(chunk_result)} chars)")

                            cleaned_text = "\n\n".join(chunks)
                            if status:
                                st.write(f"✨ All {num_chunks} chunks merged! Final document: {len(cleaned_text)} chars")

                    except Exception as e:
                        # If AI structuring fails, return the cleaned text as-is
                        print(f"AI structuring failed: {e}")
                        pass
            else:
                # For very large documents, add basic structure manually
                # Split by double newlines and add basic headings
                paragraphs = cleaned_text.split('\n\n')
                if len(paragraphs) > 5:
                    # Add a title at the beginning
                    cleaned_text = f"# Document\n\n{cleaned_text}"

        progress_bar.progress(1.0, text="Conversion complete!")

        return cleaned_text
    finally:
        # Always clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def convert_to_markdown_timed(uploaded_file, progress_bar, status=None):
    start_time = time.time()

    # Existing conversion logic
    md_text = convert_to_markdown_with_progress(uploaded_file, progress_bar, status=status)

    end_time = time.time()
    duration = end_time - start_time
    return md_text, duration

def parse_markdown_to_tree_timed(md_text):
    start_time = time.time()
    
    # Existing parsing logic
    tree = parse_markdown_to_tree(md_text)
    
    end_time = time.time()
    duration = end_time - start_time
    return tree, duration

import os
import streamlit as st
from sentence_transformers import SentenceTransformer

# Define where you want the model to live on your disk
MODEL_PATH = "./local_all_minilm"

@st.cache_resource
def get_embedding_model():
    # Check if our specific local folder already has the model
    if not os.path.exists(MODEL_PATH):
        # This triggers the initial download from Hugging Face
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Save it so we never have to download again
        model.save(MODEL_PATH)
        st.write("✅ Model downloaded and saved to disk.")
    
    # Load from the local path
    return SentenceTransformer(MODEL_PATH)

def generate_rag_answer(query, top_k_results, model_name="mistral-large-latest"):
    """
    Generate answer using RAG with top-k retrieved contexts.

    Args:
        query: User's question
        top_k_results: List of (node, score) tuples from retrieval
        model_name: Mistral model to use

    Returns:
        Generated answer string
    """
    from mistralai import Mistral

    # Extract contexts from results
    contexts = []
    for node, score in top_k_results:
        context_text = f"**{node.title}**\n{node.content}"
        contexts.append(context_text)

    # Combine contexts
    combined_context = "\n\n---\n\n".join(contexts)

    # Create prompt
    prompt = f"""You are a helpful assistant answering questions based on the provided document sections.

Use ONLY the information from the following document sections to answer the question. If the answer is not in the provided sections, say so.

Document Sections:
{combined_context}

Question: {query}

Answer: Provide a clear, accurate answer based only on the information above. Cite which section(s) you used."""

    # Call Mistral API
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "Error: MISTRAL_API_KEY not found in environment variables."

    client = Mistral(api_key=api_key)

    messages = [{"role": "user", "content": prompt}]

    response = client.chat.complete(
        model=model_name,
        messages=messages
    )

    return response.choices[0].message.content

def evaluate_with_ragas(query, answer, contexts, ground_truth=None):
    """
    Evaluate RAG response using RAGAS metrics.

    Args:
        query: User's question
        answer: Generated answer
        contexts: List of context strings used
        ground_truth: Optional reference answer

    Returns:
        Dictionary of evaluation scores
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OPENAI_API_KEY not found in environment variables. Please add it to your .env file."}

        # Create OpenAI LLM for judging
        llm = ChatOpenAI(
            model="gpt-5-mini",
            api_key=api_key,
            temperature=0
        )

        # Create OpenAI embeddings for RAGAS metrics
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key
        )

        # Prepare data for RAGAS
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }

        # Add ground truth if provided
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        else:
            # Without ground truth, use metrics that don't require it
            metrics = [faithfulness, answer_relevancy]

        dataset = Dataset.from_dict(data)

        # Evaluate with both LLM and embeddings
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )

        return result

    except Exception as e:
        return {"error": str(e)}

