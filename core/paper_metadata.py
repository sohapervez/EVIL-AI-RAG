"""Paper metadata extraction: titles, authors, and query classification.

Extracted from app1.py -- these functions have no UI dependencies.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import config
from core.retriever import _get_chroma_client

logger = logging.getLogger(__name__)


def extract_paper_titles() -> str:
    """Look up the first page of each indexed paper in ChromaDB and
    extract the real title from the content.

    Also checks which PDFs exist in the folder but failed to index.

    Returns a formatted string like:
        1. "Actual Paper Title" (file: 2311.18440v1.pdf)
        2. "Another Title" (file: paper2.pdf)
        3. "Failed to Index" (file: broken.pdf) [NOT INDEXED]
    """
    try:
        # Get list of all PDFs in the folder
        papers_dir = Path(config.DATA_DIR)
        pdf_files = sorted(papers_dir.glob("*.pdf"))
        all_pdf_names = {f.name for f in pdf_files}

        client = _get_chroma_client()
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
            title = title_from_content(content)
            authors = extract_authors_from_content(content)
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


def title_from_content(content: str) -> str:
    """Extract the paper title from the first page content.

    Strategy:
    - SKIP lines that are clearly metadata (conference headers, DOIs, etc.)
    - STOP at lines that come AFTER the title (authors, affiliations, abstract)
    - COLLECT everything else as the title (usually 1 line)
    """
    lines = content.strip().split("\n")
    title_parts: list[str] = []
    for line in lines:
        stripped = line.strip().strip("#").strip()
        if not stripped or len(stripped) < 5:
            continue
        lower = stripped.lower()

        # SKIP: metadata lines that appear BEFORE the title -- keep looking
        skip_patterns = [
            "proceedings", "conference on", "workshop on", "symposium on",
            "journal of", "transactions on", "lecture notes", "lncs",
            "vol.", "volume", "doi:", "arxiv", "http", "https",
            "published", "accepted", "received", "copyright", "\u00a9",
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
        if re.fullmatch(r"[\d\s,.\-\u2013]+", stripped):
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

        # STOP: author lines -- names separated by commas (only if we already have a title)
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


def extract_authors_from_content(content: str) -> str:
    """Extract author names from the first page content.

    Authors typically appear after the title and before affiliations/abstract.
    Returns a comma-separated string of author names, or empty string if not found.
    """
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
        cleaned_text = re.sub(r'[\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2070]', '', cleaned_text)  # Remove superscripts
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

        # Pattern 2: Names with "de" - "Jose Siqueira de Cerqueira"
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


def extract_paper_list() -> list[dict]:
    """Return structured paper metadata as a list of dicts.

    Each dict has keys: number, title, filename, authors, indexed.
    Used by the API for the WordPress plugin.
    """
    paper_info = extract_paper_titles()
    papers = []
    for line in paper_info.split("\n"):
        if not line.strip():
            continue
        if "[FAILED TO INDEX]" in line:
            m = re.search(r'\(file:\s+([^)]+)\)', line)
            if m:
                papers.append({
                    "title": "[FAILED TO INDEX]",
                    "filename": m.group(1),
                    "authors": "",
                    "indexed": False,
                })
            continue
        m = re.search(r'(\d+)\.\s+"([^"]+)"\s+\(file:\s+([^)]+)\)(?:\s+\[Authors:\s+([^\]]*)\])?', line)
        if m:
            papers.append({
                "title": m.group(2),
                "filename": m.group(3),
                "authors": m.group(4) or "",
                "indexed": True,
            })
    return papers


def enhance_query_with_paper_info(query: str, paper_info: str) -> tuple[str, bool]:
    """Enhance query by mapping paper numbers to actual paper titles/filenames.

    Example: "summary of 3rd paper" -> "summary of Specifying Fairness ..."
    This helps retrieval find relevant content.

    Returns (enhanced_query, is_general_hint).
    """
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


def is_general_question(q: str) -> bool:
    """Return True if the query is a general/cross-paper question."""
    ql = q.lower()
    general_phrases = [
        "overall", "in general", "across the papers", "across papers", "across all",
        "all papers", "all of the papers", "the literature", "what do the papers say",
        "what does the research say", "summarize the research", "key themes",
        "compare", "comparison", "common", "differences", "trend", "trends",
        "what are the main", "what are the key", "synthesize",
    ]
    return any(p in ql for p in general_phrases)


def is_paper_specific_question(q: str, paper_info: str) -> str | None:
    """Detect whether the query targets a specific paper.

    Returns the filename of the matched paper, or None.
    """
    if not paper_info:
        return None

    ql = q.lower()

    papers = []
    for line in paper_info.split("\n"):
        m = re.search(r'(\d+)\.\s+"([^"]+)"\s+\(file:\s+([^)]+)\)', line)
        if m:
            num, title, filename = m.groups()
            papers.append((int(num), title, filename))

    # 1) filename explicitly mentioned -> strongest signal
    for _, _, fn in papers:
        if fn.lower() in ql:
            return fn

    # 2) "paper 3" / "3rd paper" etc. -> strong signal
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
        "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "been", "have", "has", "had",
        "of", "a", "an", "in", "on", "at", "to", "by", "as", "into", "across"
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
    # 1) If we have 2+ shared meaningful tokens AND it's clearly better than runner-up -> accept.
    if best_common >= 2 and (best_overlap - second_overlap) >= 0.08:
        return best_fn

    # 2) If we have 3+ shared tokens, accept even if overlap ratio is modest.
    if best_common >= 3 and (best_overlap - second_overlap) >= 0.05:
        return best_fn

    return None
