"""
Phase-0

Build Index app (WebApp 0)
- Via SearchArtWorks(): quering the data source, pre-processes it.
- Via ArtEmbedd():      gets respective embeddings, L2-normalises.
- Via ArtIndexer():     create_or_append_index() - .index  .json
"""
import io
import os

import numpy as np
import requests

from PIL import Image

from ArtEmbedd import ArtEmbedd
from ArtIndexer import ArtIndexer
from SearchArtWorks import SearchArtWorks

import streamlit as st

from config import get_config
WIKI_HEADERS = get_config("WIKI_HEADERS")
TIMEOUT      = get_config("TIMEOUT")
INDEX_FILE   = get_config("INDEX_FILE")

META_FILE    = get_config("META_FILE")
SK_INDEX     = get_config("SK_INDEX")
SK_META      = get_config("SK_META")

# faced errors,
IMAGE_TIMEOUT = max(int(TIMEOUT or 10), 50)

st.set_page_config(page_title="Build Index", layout="wide")
SEARCH_MODE = ["Free Text Fetch", "Exact QIDs List"]

@st.cache_resource(show_spinner="Loading model…")
def get_tools():
    return SearchArtWorks(), ArtIndexer(INDEX_FILE, META_FILE), ArtEmbedd()


searcher, indexer, embedder = get_tools()

page = st.radio("Navigate", ["Build Index"], index=0, horizontal=True)


if page == "Build Index":
    st.title("Build Index")
    search_type = st.radio("Fetch Mode", SEARCH_MODE, horizontal=True)
    is_qid = (search_type == SEARCH_MODE[1])

    # guide to user
    text_guide = "e.g., Q17334997 Q20267558 Q18599951" if is_qid else "e.g., gothic landscape"
    search_bar = "exact space sepearted QID values" if is_qid else "to fetch by atleast one key-word"
    search_phrase = st.text_input(f"Enter {search_bar}", value="", placeholder=text_guide)
    
    min_val, max_val, default_val = (5, 20, 10)
    limit = st.slider("Limit", min_value=min_val, max_value=max_val, value=default_val)

    if st.button("Fetch & Build"):
        if not search_phrase.strip():
            st.warning("Enter a search phrase.")
            st.stop()

        #SPARQL fetch
        with st.spinner("Data Querying…"):
            try:
                raw = searcher.search_wikidata_artworks(search_phrase, limit, is_qid_search=is_qid)
                df  = searcher.process_sparql_results(raw)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        if df.empty:
            st.warning("No results found.")
            st.stop()

        st.caption(f"returned data, {len(df)} records — starting embedding loop.")

        embeddings  = []
        records     = []
        errors      = []
        status_box  = st.empty()
        prog        = st.progress(0, text="Loading…")


        for i, row in df.iterrows():
            row_id  = row.get("id", str(i))
            row_title = row.get("title", "")
            url     = row.get("image_url", "")

            if not url:
                errors.append(f"[Skip — no URL]  {row_id} {row_title}")
                prog.progress((i + 1) / len(df))
                continue

            try:
                # image fetchig
                resp = requests.get( url, timeout=IMAGE_TIMEOUT,
                            headers=WIKI_HEADERS, allow_redirects=True,
                    stream=True,          # avoids loading huge files
                )
                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    
                    errors.append(f"[Skip — not image ({content_type})]  {row_id} {row_title}")
                    prog.progress((i + 1) / len(df))
                    continue

                img = Image.open(io.BytesIO(resp.content)).convert("RGB")


                emb_img = embedder.get_embedding(img)

                # meta-data store prep
                title   = str(row.get("title")   or "Untitled")
                artist  = str(row.get("artist")  or "Unknown Artist")
                description = str(row.get("description") or "")

                depicts = str(row.get("depicts") or "")


                embeddings.append(emb_img)
                records.append({
                        "qid":         row_id,  "image_url":   url,  "title":       title,
                    "artist":      artist, "description": description, "depicts":     depicts,
                    "wikidata_url": str(row.get("wikidata_url") or ""),
                })
                status_box.caption(f"[ Done ]  {row_id} — {title}")

            except requests.exceptions.Timeout:
                errors.append(f"[Timeout]  {row_id} {row_title}")
                status_box.caption(f"[ Timeout ]  {row_id} — {row_title}")
            
            except requests.exceptions.HTTPError as exc:
                errors.append(f"[HTTP {exc.response.status_code}]  {row_id} {row_title}")
                status_box.caption(f"[ HTTP error ]  {row_id} — {row_title} : {exc}")
            except Exception as exc:
                errors.append(f"[Error]  {row_id} {row_title} : {exc}")
                status_box.caption(f"[ Error ]  {row_id} — {row_title} : {exc}")

            prog.progress((i + 1) / len(df), text=f"Loading {i + 1}/{len(df)}") #Embedding


        st.caption(f"Successfully loaded: {len(embeddings)} / {len(df)}")
        if errors:
            with st.expander(f"Skipped / failed ({len(errors)})"):
                for e in errors:
                    st.text(e)

        if not embeddings:
            st.error("No embeddings could be created — check errors above.")
            st.stop()


        with st.spinner("Building index…"):
            
            index, meta = indexer.create_or_append_index(embeddings, records)

        st.session_state[SK_INDEX] = index
        st.session_state[SK_META]  = meta
        
        index_built = f", overall_total **{index.ntotal}** vectors " if index.ntotal else "" 

        st.success( f"After embedding, Index built {index_built}" + " & saved to `{INDEX_FILE}`")


