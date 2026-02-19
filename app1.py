"""EVIL-AI Research Paper RAG Chatbot.

A straightforward Q&A chatbot where users ask questions and get answers
based on research papers (PDFs) stored in ``data/papers/``.
The user drives the conversation — no follow-up questions are asked.

Usage:
    streamlit run app1.py
"""

# --- 1. Setup & Imports ---
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

import config
from core.rag_chain import ChatMessage, query_stream

logger = logging.getLogger(__name__)

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="EVIL-AI Research Q&A",
    page_icon="📚",
)

# --- 3. Session State Defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "auto_indexed" not in st.session_state:
    st.session_state.auto_indexed = False
if "paper_info" not in st.session_state:
    st.session_state.paper_info = ""


# --- 4. Extract Real Paper Titles from ChromaDB ---
def _extract_paper_titles() -> str:
    """Look up the first page of each indexed paper in ChromaDB and
    extract the real title from the content.

    Also checks which PDFs exist in the folder but failed to index.

    Returns a formatted string like:
        1. "Actual Paper Title" (file: 2311.18440v1.pdf)
        2. "Another Title" (file: paper2.pdf)
        3. "Failed to Index" (file: broken.pdf) [NOT INDEXED]
    """
    import chromadb

    try:
        # Get list of all PDFs in the folder
        papers_dir = Path(config.DATA_DIR)
        pdf_files = sorted(papers_dir.glob("*.pdf"))
        all_pdf_names = {f.name for f in pdf_files}

        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        parent_col = client.get_or_create_collection(
            name=config.CHROMA_PARENT_COLLECTION,
        )

        # Get indexed papers from ChromaDB - OPTIMIZED: fetch metadata first, then only page 1 docs
        indexed_sources: set[str] = set()
        first_pages: dict[str, str] = {}
        
        # Step 1: Get unique sources from child collection (lightweight, just metadata)
        try:
            child_col = client.get_or_create_collection(
                name=config.CHROMA_CHILD_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            if child_col.count() > 0:
                child_meta = child_col.get(include=["metadatas"])
                indexed_sources = {m.get("source", "") for m in child_meta["metadatas"] if m.get("source")}
        except Exception:
            pass
        
        # Step 2: Get metadata only from parent collection to find page 1 chunk IDs
        if parent_col.count() > 0:
            try:
                # Get only metadata first (much faster than fetching all documents)
                parent_meta = parent_col.get(include=["metadatas"])
                page1_ids: dict[str, str] = {}  # source -> chunk_id
                
                for idx, meta in enumerate(parent_meta["metadatas"]):
                    source = meta.get("source", "")
                    if source:
                        indexed_sources.add(source)
                        page = meta.get("page", 0)
                        try:
                            page = int(page)
                        except (ValueError, TypeError):
                            page = 0
                        if page == 1 and source not in page1_ids:
                            # Store the index to fetch this document later
                            page1_ids[source] = parent_meta["ids"][idx]
                
                # Step 3: Fetch only the page 1 documents we need (much smaller)
                if page1_ids:
                    needed_ids = list(page1_ids.values())
                    page1_data = parent_col.get(ids=needed_ids, include=["documents", "metadatas"])
                    # Map back to sources
                    id_to_source = {v: k for k, v in page1_ids.items()}
                    for doc_id, doc in zip(page1_data["ids"], page1_data["documents"]):
                        if doc_id in id_to_source:
                            first_pages[id_to_source[doc_id]] = doc
            except Exception as e:
                logger.warning("Optimized title extraction failed, using fallback: %s", e)
                # Fallback: original method (slower but works)
                all_data = parent_col.get(include=["documents", "metadatas"])
                for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                    source = meta.get("source", "")
                    if source:
                        indexed_sources.add(source)
                        page = meta.get("page", 0)
                        try:
                            page = int(page)
                        except (ValueError, TypeError):
                            page = 0
                        if page == 1 and source not in first_pages:
                            first_pages[source] = doc

        # Find papers that exist but aren't indexed
        failed_papers = all_pdf_names - indexed_sources

        lines_out: list[str] = []
        # Add successfully indexed papers
        for i, (source, content) in enumerate(sorted(first_pages.items()), 1):
            title = _title_from_content(content)
            authors = _extract_authors_from_content(content)
            if authors:
                lines_out.append(f'{i}. "{title}" (file: {source}) [Authors: {authors}]')
            else:
                lines_out.append(f'{i}. "{title}" (file: {source})')

        # Add failed papers at the end
        for failed_paper in sorted(failed_papers):
            lines_out.append(f'{len(lines_out) + 1}. "[FAILED TO INDEX]" (file: {failed_paper})')

        return "\n".join(lines_out) if lines_out else ""
    except Exception as e:
        logger.warning("Could not extract paper titles: %s", e)
        return ""


def _title_from_content(content: str) -> str:
    """Extract the paper title from the first page content.

    Strategy:
    - SKIP lines that are clearly metadata (conference headers, DOIs, etc.)
    - STOP at lines that come AFTER the title (authors, affiliations, abstract)
    - COLLECT everything else as the title (usually 1 line)
    """
    import re

    lines = content.strip().split("\n")
    title_parts: list[str] = []
    for line in lines:
        stripped = line.strip().strip("#").strip()
        if not stripped or len(stripped) < 5:
            continue
        lower = stripped.lower()

        # SKIP: metadata lines that appear BEFORE the title — keep looking
        skip_patterns = [
            "proceedings", "conference on", "workshop on", "symposium on",
            "journal of", "transactions on", "lecture notes", "lncs",
            "vol.", "volume", "doi:", "arxiv", "http", "https",
            "published", "accepted", "received", "copyright", "©",
            "springer", "isbn", "issn", "acm", "ieee",
            "pp.", "pages ",
        ]
        if any(kw in lower for kw in skip_patterns):
            continue

        # SKIP: date/location lines (e.g. "June 4-7, 2024, City, Country")
        if re.search(
            r"(january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d",
            lower,
        ):
            continue

        # SKIP: lines that are just years or short numbers
        if re.fullmatch(r"[\d\s,.\-–]+", stripped):
            continue

        # STOP: abstract or email (these come AFTER the title and authors)
        if any(kw in lower for kw in ["abstract", "@"]):
            break

        # STOP: affiliation lines (come after authors, so title is already found)
        if any(kw in lower for kw in [
            "university", "department", "institute", "school of",
            "faculty of", "college of",
        ]):
            break

        # STOP: author lines — names separated by commas (only if we already have a title)
        # Be more careful: titles can have commas, so we need better detection
        if title_parts:
            # Author lines typically have: "FirstName LastName, FirstName LastName, ..."
            # Pattern: multiple "FirstName LastName" patterns separated by commas
            name_pattern = r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
            name_matches = re.findall(name_pattern, stripped)
            
            # If we see 2+ name patterns AND multiple commas, it's likely authors
            if stripped.count(",") >= 2 and len(name_matches) >= 2:
                break
            
            # Also stop if line looks like just names without title-like words
            # (e.g., "John Smith and Jane Doe" without context words)
            if len(name_matches) >= 2 and not any(word in lower for word in [
                "requirements", "allocation", "fairness", "transparency", 
                "ethics", "agents", "ai", "for", "in", "of", "the", "a"
            ]):
                break

        # This line looks like part of the title
        title_parts.append(stripped)

        # Allow multi-line titles (up to 4 lines max to avoid grabbing too much)
        # Most titles are 1-2 lines, but some can be longer
        if len(title_parts) >= 4:
            break

    # Join all title parts with spaces to form the complete title
    return " ".join(title_parts) if title_parts else "Unknown Title"


def _extract_authors_from_content(content: str) -> str:
    """Extract author names from the first page content.
    
    Authors typically appear after the title and before affiliations/abstract.
    Returns a comma-separated string of author names, or empty string if not found.
    """
    import re
    
    lines = content.strip().split("\n")
    title_found = False
    author_lines: list[str] = []
    
    for line in lines:
        stripped = line.strip().strip("#").strip()
        if not stripped or len(stripped) < 3:
            continue
        lower = stripped.lower()
        
        # Skip metadata before title
        if any(kw in lower for kw in [
            "proceedings", "conference", "workshop", "symposium",
            "journal", "volume", "doi", "arxiv", "http",
            "published", "copyright", "springer", "isbn",
        ]):
            continue
        
        # Mark when we've passed the title (title extraction logic)
        # If we see title-like content, mark title_found
        if not title_found:
            # Check if this could be part of title (has title-like words)
            if any(word in lower for word in [
                "requirements", "allocation", "fairness", "transparency",
                "ethics", "agents", "ai", "specifying", "operationalizing",
            ]) or len(stripped) > 20:
                # This might be title, continue to next line
                continue
            # If we see author-like patterns, title was likely before this
            if re.search(r"\b[A-Z][a-z]+\s+[A-Z]", stripped):
                title_found = True
        
        # After title, look for author lines
        if title_found:
            # Skip if it's abstract or email
            if any(kw in lower for kw in ["abstract", "@"]):
                break
            
            # Skip affiliation lines
            if any(kw in lower for kw in [
                "university", "department", "institute", "school",
                "faculty", "college", "laboratory", "lab",
            ]):
                # Affiliations come after authors, so we're done
                break
            
            # Author lines typically have:
            # - Names with "FirstName LastName" pattern
            # - Separated by commas or "and"
            # - May have superscripts/numbers for affiliations
            # - May have ORCID IDs in brackets
            
            # Check if this looks like author names BEFORE cleaning
            # Pattern: "FirstName LastName" or "FirstName Middle LastName"
            name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
            names = re.findall(name_pattern, stripped)
            
            if names and len(names) >= 1:
                # This looks like an author line - collect it
                author_lines.append(stripped)
            
            # Collect up to 3 lines (authors usually span 1-2 lines, max 3)
            if len(author_lines) >= 3:
                break
    
    # Join all author lines and extract individual names
    if author_lines:
        # Join lines with space (handles multi-line author lists)
        all_authors_text = " ".join(author_lines)
        
        # First, clean the text: remove ORCID IDs, superscripts, and digits
        cleaned_text = re.sub(r'\[.*?\]', '', all_authors_text)  # Remove [ORCID]
        cleaned_text = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]', '', cleaned_text)  # Remove superscripts
        cleaned_text = re.sub(r'\b\d+\b', '', cleaned_text)  # Remove standalone numbers
        cleaned_text = cleaned_text.strip()
        
        # Extract author names using comprehensive pattern matching
        # Try multiple patterns to catch all name formats
        author_names = []
        
        # Pattern 1: Names with initials like "F.C." - "Amanda Aline F.C. Vicenzi"
        pattern1 = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.[A-Z]\.?)(?:\s+[A-Z][a-z]+)+\b'
        matches1 = re.findall(pattern1, cleaned_text)
        for m in matches1:
            clean_m = " ".join(m.split())
            if clean_m and clean_m not in author_names:
                author_names.append(clean_m)
        
        # Pattern 2: Names with "de" - "José Siqueira de Cerqueira"
        pattern2 = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+de\s+[A-Z][a-z]+)+\b'
        matches2 = re.findall(pattern2, cleaned_text)
        for m in matches2:
            clean_m = " ".join(m.split())
            if clean_m and clean_m not in author_names:
                author_names.append(clean_m)
        
        # Pattern 3: Simple names (2-4 words) - "Pekka Abrahamsson", "Edna Dias Canedo"
        # Only use this if we haven't found enough authors yet
        if len(author_names) < 4:
            pattern3 = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b'
            matches3 = re.findall(pattern3, cleaned_text)
            for m in matches3:
                clean_m = " ".join(m.split())
                # Only add if not already in list and not a substring of existing names
                if clean_m not in author_names and len(clean_m.split()) >= 2:
                    # Check if it's a substring of an existing longer name
                    is_part_of_existing = any(clean_m in name for name in author_names if len(name) > len(clean_m))
                    if not is_part_of_existing:
                        author_names.append(clean_m)
        
        # If pattern matching didn't find enough, try splitting by comma/and and extracting
        if len(author_names) < 2:
            parts = re.split(r'\s*,\s*|\s+and\s+', cleaned_text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # Extract capitalized word sequences (likely names)
                words = [w.strip('.,;') for w in part.split() if w.strip('.,;')]
                # Find sequences of capitalized words (2-5 words)
                name_parts = []
                for word in words:
                    if word and word[0].isupper() and word.lower() not in ["and", "et", "al", "the"]:
                        name_parts.append(word)
                        if len(name_parts) >= 2:
                            potential_name = " ".join(name_parts)
                            if potential_name not in author_names and len(potential_name.split()) >= 2:
                                author_names.append(potential_name)
                    elif len(name_parts) >= 2:
                        # End of name sequence
                        name_parts = []
                # Don't forget last sequence
                if len(name_parts) >= 2:
                    potential_name = " ".join(name_parts)
                    if potential_name not in author_names:
                        author_names.append(potential_name)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_authors = []
        for name in author_names:
            # Normalize: remove extra spaces
            name = " ".join(name.split())
            if name and name.lower() not in seen and len(name.split()) >= 2:
                seen.add(name.lower())
                unique_authors.append(name)
        
        if unique_authors:
            return ", ".join(unique_authors)
    
    return ""


def _enhance_query_with_paper_info(query: str, paper_info: str) -> str:
    """Enhance query by mapping paper numbers to actual paper titles/filenames.
    
    Example: "summary of 3rd paper" -> "summary of Specifying Fairness and Transparency Requirements for Public Benefit Allocation"
    This helps retrieval find relevant content.
    """
    import re
    
    if not paper_info:
        return query, False
    
    # Parse paper info to extract titles and filenames
    papers = []
    for line in paper_info.split("\n"):
        if not line.strip() or "[FAILED TO INDEX]" in line:
            continue
        # Extract paper number, title, and filename
        # Format: "1. "Title" (file: filename.pdf) [Authors: ...]"
        match = re.search(r'(\d+)\.\s+"([^"]+)"\s+\(file:\s+([^)]+)\)', line)
        if match:
            num, title, filename = match.groups()
            papers.append({
                "number": int(num),
                "title": title,
                "filename": filename,
            })
    
    if not papers:
        return query, False
    
    # Map number words to numbers
    number_map = {
        "first": 1, "1st": 1, "one": 1,
        "second": 2, "2nd": 2, "two": 2,
        "third": 3, "3rd": 3, "three": 3,
        "fourth": 4, "4th": 4, "four": 4,
        "fifth": 5, "5th": 5, "five": 5,
    }
    
    query_lower = query.lower()
    enhanced = query
    
    # Check if query mentions a paper number
    for word, num in number_map.items():
        if word in query_lower and num <= len(papers):
            paper = papers[num - 1]  # Convert to 0-indexed
            # Add paper title and filename to query for better retrieval
            enhanced = f"{query} {paper['title']} {paper['filename']}"
            break
    
    # Check if query mentions a specific paper title (add filename for better retrieval)
    if enhanced == query:  # Only if not already enhanced by paper number
        for paper in papers:
            title_lower = paper['title'].lower()
            # Normalize title for matching
            normalized_title = re.sub(r'[^\w\s-]', ' ', title_lower)
            title_words = [w.strip() for w in re.split(r'[\s-]+', normalized_title) if len(w.strip()) > 2]
            stop_words = {"the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "been", "have", "has", "had"}
            title_words = [w for w in title_words if w not in stop_words]
            
            # Check for matches
            matches = sum(1 for w in title_words if w in query_lower)
            min_matches = max(2, min(3, len(title_words) // 3))
            
            if matches >= min_matches:
                # Add filename to query to help retrieval find the right document
                enhanced = f"{query} {paper['filename']}"
                break
    
    # Detect general research questions that should cover ALL papers
    general_phrases = [
        "all papers", "all work", "everything", "general summary", "overview",
        "what research", "research done", "research has been", "research under",
        "what has been", "what work", "work done", "projects", "studies",
        "overall", "in general", "summarize all", "all research"
    ]
    
    if any(phrase in query_lower for phrase in general_phrases):
        # Add all paper titles to ensure retrieval from all papers
        all_titles = " ".join([p["title"] for p in papers])
        enhanced = f"{query} {all_titles}"
        # Mark as general query for later processing
        return enhanced, True
    
    return enhanced, False

def _is_general_question(q: str) -> bool:
    ql = q.lower()
    general_phrases = [
        "overall", "in general", "across the papers", "across papers", "across all",
        "all papers", "all of the papers", "the literature", "what do the papers say",
        "what does the research say", "summarize the research", "key themes",
        "compare", "comparison", "common", "differences", "trend", "trends",
        "what are the main", "what are the key", "synthesize",
    ]
    return any(p in ql for p in general_phrases)


def _is_paper_specific_question(q: str, paper_info: str) -> str | None:
    import re

    if not paper_info:
        return None

    ql = q.lower()

    papers = []
    for line in paper_info.split("\n"):
        m = re.search(r'(\d+)\.\s+"([^"]+)"\s+\(file:\s+([^)]+)\)', line)
        if m:
            num, title, filename = m.groups()
            papers.append((int(num), title, filename))

    # 1) filename explicitly mentioned → strongest signal
    for _, _, fn in papers:
        if fn.lower() in ql:
            return fn

    # 2) "paper 3" / "3rd paper" etc. → strong signal
    number_map = {
        "first": 1, "1st": 1, "one": 1,
        "second": 2, "2nd": 2, "two": 2,
        "third": 3, "3rd": 3, "three": 3,
        "fourth": 4, "4th": 4, "four": 4,
        "fifth": 5, "5th": 5, "five": 5,
        "sixth": 6, "6th": 6, "six": 6,
        "seventh": 7, "7th": 7, "seven": 7,
        "eighth": 8, "8th": 8, "eight": 8,
    }
    for word, n in number_map.items():
        if word in ql or f"paper {n}" in ql:
            for num, _, fn in papers:
                if num == n:
                    return fn

    # 3) strict title-match scoring (prevents wrong paper selection)
    stop_words = {
        "the","and","for","with","from","that","this","are","was","were","been","have","has","had",
        "of","a","an","in","on","at","to","by","as","into","across"
    }

    def norm_tokens(text: str) -> set[str]:
        text = re.sub(r"[^\w\s-]", " ", text.lower())
        parts = re.split(r"[\s-]+", text)
        toks = {p.strip() for p in parts if len(p.strip()) > 2 and p.strip() not in stop_words}
        return toks

    q_tokens = norm_tokens(ql)

    scored = []
    for _, title, fn in papers:
        t_tokens = norm_tokens(title)
        if not t_tokens:
            continue
        common = len(q_tokens.intersection(t_tokens))
        overlap = common / max(len(t_tokens), 1)
        scored.append((overlap, common, fn))

    if not scored:
        return None

    scored.sort(reverse=True)  # highest overlap first
    best_overlap, best_common, best_fn = scored[0]
    second_overlap = scored[1][0] if len(scored) > 1 else 0.0

    ## Require evidence + avoid ambiguous close matches
    # Long titles often produce lower overlap ratios, so we use a more forgiving rule.
    # 1) If we have 2+ shared meaningful tokens AND it's clearly better than runner-up → accept.
    if best_common >= 2 and (best_overlap - second_overlap) >= 0.08:
        return best_fn

    # 2) If we have 3+ shared tokens, accept even if overlap ratio is modest.
    if best_common >= 3 and (best_overlap - second_overlap) >= 0.05:
        return best_fn
    
    return None


# --- 5. Auto-Index PDFs on First Load ---
def _sync_papers() -> str:
    """Index any new PDFs found in data/papers/ into ChromaDB."""
    import chromadb
    from ingest import ingest_directory

    papers_dir = Path(config.DATA_DIR)
    papers_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        return "No PDFs found in `data/papers/`."

    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        child_col = client.get_or_create_collection(
            name=config.CHROMA_CHILD_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        if child_col.count() > 0:
            all_meta = child_col.get(include=["metadatas"])
            already = {m.get("source", "") for m in all_meta["metadatas"]}
        else:
            already = set()
    except Exception:
        already = set()

    new_files = [f for f in pdf_files if f.name not in already]
    if not new_files:
        return f"{len(pdf_files)} paper(s) already indexed."

    summary = ingest_directory(pdf_dir=papers_dir, clear=False)

    if summary["files"] == 0:
        return f"{len(pdf_files)} paper(s) already indexed."

    return (
        f"Indexed **{summary['files']}** new paper(s): "
        f"{summary['parents']} parent chunks, {summary['children']} child chunks."
    )


if not st.session_state.auto_indexed:
    with st.spinner("Indexing research papers..."):
        status_msg = _sync_papers()
    st.session_state.paper_info = _extract_paper_titles()
    st.session_state.auto_indexed = True
    if "No PDFs" in status_msg:
        st.warning(status_msg)
    elif "already indexed" not in status_msg:
        st.toast(status_msg, icon="✅")
    
    # Check for failed papers and show warning with specific file names
    if "[FAILED TO INDEX]" in st.session_state.paper_info:
        # Extract failed file names from paper_info
        failed_files = [
            line.split("(file: ")[1].split(")")[0]
            for line in st.session_state.paper_info.split("\n")
            if "[FAILED TO INDEX]" in line
        ]
        failed_list = ", ".join(failed_files)
        st.warning(
            f"⚠️ **Some papers failed to index** (PDF parsing errors):\n\n"
            f"**Failed files:** {failed_list}\n\n"
            f"These papers are listed but won't be searchable. "
            f"The PDFs may be corrupted or have unsupported formatting. "
            f"Try re-saving the PDFs or converting them to a standard format."
        )

if not st.session_state.paper_info:
    st.session_state.paper_info = _extract_paper_titles()
    if "[FAILED TO INDEX]" in st.session_state.paper_info:
        failed_files = [
            line.split("(file: ")[1].split(")")[0]
            for line in st.session_state.paper_info.split("\n")
            if "[FAILED TO INDEX]" in line
        ]
        failed_list = ", ".join(failed_files)
        st.warning(
            f"⚠️ **Some papers failed to index** (PDF parsing errors):\n\n"
            f"**Failed files:** {failed_list}\n\n"
            f"These papers are listed but won't be searchable."
        )


# --- 6. System Prompt ---
SYSTEM_PROMPT = f"""You are a research assistant for the EVIL-AI project. \
Your role is to answer questions about the research papers by synthesizing information from the provided context.

The EVIL-AI project has these research papers:
{st.session_state.paper_info}

IMPORTANT INSTRUCTIONS:

GENERAL ANSWERING RULES:
- Answer ONLY using the provided context from the research papers.
- Do NOT use outside knowledge.
- If the context does not contain enough information, clearly say so.
- Cite sources as [Source: filename, Page X].

PAPER-SPECIFIC QUESTIONS:
- If the question clearly refers to a specific paper (by title, filename, or paper number),
  you MUST restrict your answer to that paper only.
- Do NOT include information from other papers unless the user explicitly asks to compare or combine them.
- If retrieved context includes chunks from unrelated papers, ignore them.

GENERAL OR CROSS-PAPER QUESTIONS:
- Only synthesize across multiple papers when:
  • The user explicitly asks about multiple papers, OR
  • The user asks about overall EVIL-AI research, OR
  • The question is clearly comparative.
- When synthesizing across papers, include only papers for which relevant context was retrieved.
- Do NOT force inclusion of every paper unless the question explicitly requires all papers.

SUMMARIES:
- When asked for a summary, abstract, or overview of a specific paper,
  synthesize key points from that paper only (e.g., introduction, methodology, findings, conclusions).
- When asked for an overall project summary,
  synthesize across relevant papers and clearly indicate which findings come from which paper.

CONTEXT DISCIPLINE:
- If multiple filenames appear in the retrieved context, do NOT automatically use all of them.
- Use only the context that directly answers the question.
- Ignore context that is unrelated to the question, even if retrieved.

ACCURACY PRIORITY:
- Prefer precision over breadth.
- Do not speculate.
- Do not infer beyond what is explicitly supported by the context.

CRITICAL AUTHOR RULES:
- NEVER mention author names in summaries, discussions, or content answers
  UNLESS the user explicitly asks about authors.
- When discussing paper content, refer ONLY to the exact paper titles listed above.
- Author names should appear ONLY when the question specifically asks for them.
- If author information appears in the retrieved context, ignore it unless the question is about authors.

PAPER TITLE RULES:
- When referring to papers, use ONLY the exact titles listed above.
- Never invent or modify titles.
- When asked to list paper titles, list ALL papers shown above,
  including those marked "[FAILED TO INDEX]".
- When listing titles, do NOT include authors or affiliations.

AUTHOR LISTING RULES:
- When asked about authors, use ONLY the author information provided in the paper list above.
- Do NOT include ORCID IDs, brackets, superscripts, or digits.
- Example correct format:
  "Amanda Aline F.C. Vicenzi, José Siqueira de Cerqueira, Pekka Abrahamsson, Edna Dias Canedo"

FAILURE CONDITION:
- Only say "I don't have that information in the provided context"
  if the context truly contains no relevant information at all."""


# --- 7. Title ---
st.title("📚 EVIL-AI Research Q&A")

# --- 8. Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📖 Sources ({len(msg['sources'])} references)"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']}** — Page {src['page']}, "
                        f"Section: {src['section']} ({src['content_type']})"
                    )
                    st.caption(
                        src["content"][:300]
                        + ("..." if len(src["content"]) > 300 else "")
                    )
                    st.divider()

# --- 9. Chat Input & Response ---
if user_input := st.chat_input("Ask a question about the research papers..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation history
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in st.session_state.messages[:-1]
    ]

    # Stream the answer
    with st.chat_message("assistant"):
        try:
            # First, check if user is asking about a specific paper (use original query)
            target_source = _is_paper_specific_question(
                user_input,  # Use original query for title detection
                st.session_state.paper_info
            )
            
            # Then enhance query: map paper numbers to titles/filenames
            enhanced_query, general_hint = _enhance_query_with_paper_info(
                user_input,
                st.session_state.paper_info
            )
            
            # If we detected a specific paper but enhancement didn't add filename, add it now
            if target_source and target_source not in enhanced_query:
                enhanced_query = f"{enhanced_query} {target_source}"

            # If no clear paper specified → treat as general query
            is_general_query = (
                general_hint
                or _is_general_question(enhanced_query)
                or (target_source is None)
            )

            # Apply filter only for clearly paper-specific queries
            filter_source = None if is_general_query else target_source

            # Set retrieval breadth based on mode
            if is_general_query:
                top_k_value = max(12, config.TOP_K * 3)
                max_tokens_value = 2048
            else:
                top_k_value = config.TOP_K
                max_tokens_value = 1024

            # Call RAG pipeline
            token_gen, sources = query_stream(
                question=enhanced_query,
                history=history,
                system_prompt=SYSTEM_PROMPT,
                llm_provider=config.LLM_PROVIDER,
                llm_model=config.LLM_MODEL,
                temperature=0.3,
                max_tokens=max_tokens_value,
                top_k=top_k_value,
                filter_source=filter_source,
                use_hybrid=config.USE_HYBRID_SEARCH,
                bm25_weight=config.BM25_WEIGHT,
                use_reranking=config.USE_RERANKING,
                embedding_provider=config.EMBEDDING_PROVIDER,
                embedding_model=config.EMBEDDING_MODEL,
            )

            full_response = st.write_stream(token_gen)

            source_dicts = []
            if sources:
                source_dicts = [
                    {
                        "source": s.source,
                        "page": s.page,
                        "section": s.section,
                        "content_type": s.content_type,
                        "content": s.content,
                    }
                    for s in sources
                ]

                with st.expander(f"📖 Sources ({len(sources)} references)"):
                    for src in source_dicts:
                        st.markdown(
                            f"**{src['source']}** — Page {src['page']}, "
                            f"Section: {src['section']} ({src['content_type']})"
                        )
                        st.caption(
                            src["content"][:300]
                            + ("..." if len(src["content"]) > 300 else "")
                        )
                        st.divider()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": source_dicts,
                }
            )

        except Exception as e:
            error_msg = f"Error: {e}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

