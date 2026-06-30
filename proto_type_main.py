"""

Expects a index from Phase-0 demo_data/, where an inital index to be built,  
Phase-1: -"Explainable-Multi-modal Search & Retrieval Pipeline"

Search app (WebApp 1).

"""
import io
import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

from SearchArtWorks import SearchArtWorks
from ArtEmbedd import ArtEmbedd
from ArtIndexer import ArtIndexer
from ExplainITQuery import ExplainITQuery
from ConceptExplainer import ConceptExplainer

import streamlit as st
from config import get_config

WIKI_HEADERS = get_config("WIKI_HEADERS")
TIMEOUT      = get_config("TIMEOUT")
INDEX_FILE   = get_config("INDEX_FILE")

META_FILE    = get_config("META_FILE")
SK_INDEX     = get_config("SK_INDEX")

SK_META      = get_config("SK_META")
FILE_EXT     = tuple(get_config("FILE_EXT"))

RADIO_MODES = ["Text description", "Upload image"]

st.set_page_config(page_title="Explainable Search", layout="wide")


@st.cache_resource(show_spinner="Loading page....")
def get_tools():
    embedd = ArtEmbedd()
    return SearchArtWorks(), ArtIndexer(INDEX_FILE, META_FILE), embedd, ExplainITQuery()

@st.cache_data(show_spinner=False, ttl=1800)  # Caching images
def get_images_batch(urls):
    import concurrent.futures
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(get_images, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            results[url] = future.result()
    return results


def get_images(img_url, TIMEOUT=TIMEOUT, WIKI_HEADERS=WIKI_HEADERS):
    try:
        
        resp = requests.get(img_url, timeout=TIMEOUT, headers=WIKI_HEADERS, 
                        stream=True, allow_redirects=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB") 
        return img
    except Exception:
        return None


searcher, indexer, embedder, explain_it_query = get_tools()


def load_index():
    """Load FAISS index and metadata; store in session state."""
    try:
        idx, meta = indexer.load_faiss_index()
        st.session_state[SK_INDEX] = idx
        st.session_state[SK_META]  = meta

        return True
    except FileNotFoundError as e:
        st.error(f"Index not found: {e}\n\nPlease build the index, contact dev team.")
        return False
    
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return False


# Attempt to load index (only once per session)
if SK_INDEX not in st.session_state or SK_META not in st.session_state:
    if not load_index():
        st.stop()   # stop execution, if index or metadata missing


# we have index and meta in session state
index = st.session_state[SK_INDEX]
meta  = st.session_state[SK_META]

concept_explainer = ConceptExplainer(embedder, index, meta)

st.title("iArt xAI Search")


mode  = st.radio("Query type", RADIO_MODES, horizontal=True)
top_k = st.slider("Top-K", 1, 10, 5)

query = None


# Query input
if mode == RADIO_MODES[0]:
    query_text = st.text_input("Query")
    if query_text.strip():
        query = query_text.strip()
    with st.expander("Keyword explanation settings", expanded=True):
        top_n_tokens = st.slider("Max keywords", 1, 10, 5)
else:
    uploaded = st.file_uploader("Upload image", type=FILE_EXT)
    if uploaded:
        query = Image.open(uploaded).convert("RGB")
        st.image(query, caption="Query image", width=300)


if query is not None and st.button("Search"):
    with st.spinner("Searching…"):
        _, results = indexer.perform_similarity_search(
            query, embedder.get_embedding, index, meta, top_k=top_k )

    # placing in session state
    st.session_state["search_query"] = query
    st.session_state["search_results"] = results

    st.session_state["search_mode"] = mode

elif query is None:
    #to clear earlier results
    st.session_state["search_query"] = None
    st.session_state["search_results"] = None
    st.session_state["search_mode"] = None


# search results display
if "search_results" in st.session_state and st.session_state["search_results"]:
    results = st.session_state["search_results"]
    query = st.session_state["search_query"]
    mode = st.session_state.get("search_mode", RADIO_MODES[0])

    # Results grid
    st.subheader(f"Top {len(results)} results")
    cols = st.columns(min(5, len(results)))

    all_urls = [r.get("image_url") for r in results if r.get("image_url")]
    image_cache = get_images_batch(all_urls)    
    for i, r in enumerate(results):
        with cols[i % len(cols)]:   
            img_url = r.get("image_url")
            if img_url:

                img_obj = image_cache.get(r.get("image_url"))
                if img_obj is not None:
                    
                    st.image(img_obj, width='stretch')
                else:
                    st.warning("Failed to download artwork.")
            else:
                st.warning("Missing image URL.")
            st.caption(f"**{r.get('title')}** | *{r.get('artist')}*")

            st.progress(  int(r.get("similarity")),
                text=f"Similarity: {r.get('similarity'):.1f}%"#  (cosine: {r.get('cosine_score'):.3f})"
            )

            depicts = r.get('depicts', '')
            if depicts:
                st.caption(f"**Depicts**: {depicts[:100]}") #TODO
            
            qid = r.get('qid', '')
            if qid:
                st.caption(f"**QID**: {qid}")  


    # Concept-level explanation from metadata ( displayed for both query types)
    st.divider()
    with st.spinner("Computing concept alignments…"):
        concept_explainer.render(query=query, results=results, top_n=5)

    st.divider()
    with st.expander("Detailed explanation for a selected result", expanded=False):

        # User may choose a result-image to explain
        result_options = [
            f"#({i+1})  {r.get('title', 'Untitled')[:40]} (sim {r.get('similarity', 0):.1f}%)"
            for i, r in enumerate(results)  ]
        selected_idx = st.selectbox(
            "Select result to explain", options=range(len(results)),
            format_func=lambda i: result_options[i] )
        top = results[selected_idx]

        st.caption( f"**{top.get('title')}** by *{top.get('artist').title()}*  |  "
            f"Similarity: {top.get('similarity'):.1f}%" )

        top_img_obj = image_cache.get(top.get("image_url"))

        st.divider()
        with st.expander("Semantic Concept similarity for the selected result", expanded=False):

            st.caption("Shows which semantic concepts are strongly present in both the Query and this artwork.")
            with st.spinner("Computing concept overlap…"):
                concept_explainer.render_single_result(query, top, n=5)
    
        st.divider()
        mode = "Keyword Importance" if isinstance(query, str) else "Region Importance Heatmap"
        with st.expander(f"{mode}", expanded=False):
            if isinstance(query, str):

                # Text query - token importance
                st.markdown("**Keyword importance (without stopwords)**")
                st.caption("The chart displays the score drop when each word is removed from the Query.")
                with st.spinner("Computing keyword importances…"):
                    token_imps = explain_it_query.explain_text_query( query, top.get("image_url"), 
                                        embedder.get_embedding, top_n=top_n_tokens )

                if token_imps:
                    col1, col2 = st.columns(2)
                    with col1:
                        
                        df_tok = pd.DataFrame(token_imps)
                                            
                        df_tok["importance_ui"] = df_tok["importance"] * 100.0
                                            
                        df_tok = df_tok.set_index("token_idx")
                        df_tok = df_tok[["importance_ui"]]
                        df_tok.columns = ["Importance (% points drop)"]
                        
                        df_tok.index = [idx.split("_", 1)[1] for idx in df_tok.index]
                        
                        st.bar_chart(df_tok, horizontal=True, y="Importance (% points drop)")

                    with col2:
                        st.image(top_img_obj, caption=f"Result Match", width="stretch")
                else:
                    st.info("Could not compute token importances for this result.")

            else:

                grid_size = st.slider( "Patch grid size for heatmap", 
                    min_value=5, max_value=8, value=6, step=1,
                    help="Larger grid gives finer resolution but slower computation." )
                # Image query- patch occlusion sensitivity
                st.markdown("**Region importance heatmap**")
                st.caption("The heatmap displays regions that most influenced the match;\
                                changing them may reduce the similarity score.")
                st.caption("Red = region that most helped the match; Blue = region that was ignored.")
                with st.spinner("Computing region importance…"):
                    result_heatmap, r_patch_metrics = explain_it_query.explain_image_query( query, top.get("image_url"), 
                                                        embedder.get_embedding, grid=grid_size  )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(query, caption="Original Query", width="stretch")
                with col2:
                    st.image(result_heatmap, caption="Region importance", width="stretch")
                    if r_patch_metrics:

                        st.caption( f"**Highest(Red) region:** {r_patch_metrics.get('highest', {}).get('importance')*100.0:.2f} % points at "
                                f"Row {r_patch_metrics.get('highest', {}).get('row')}, Col {r_patch_metrics.get('highest', {}).get('col')}"   )
                        st.caption(
                            f"**Lowest(Blue) region:** {r_patch_metrics.get('lowest', {}).get('importance')*100.0:.2f} % points at "
                            f"Row {r_patch_metrics.get('lowest', {}).get('row')}, Col {r_patch_metrics.get('lowest', {}).get('col')}"    )
                        
                with col3:
                    st.image(top_img_obj, caption="Original Retrieved result", width="stretch")
        st.divider()


