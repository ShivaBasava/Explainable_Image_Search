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

import streamlit as st
from config import get_config
WIKI_HEADERS = get_config("WIKI_HEADERS")
TIMEOUT      = get_config("TIMEOUT")
INDEX_FILE   = get_config("INDEX_FILE")

META_FILE    = get_config("META_FILE")
SK_INDEX     = get_config("SK_INDEX")

SK_META      = get_config("SK_META")
FILE_EXT     = tuple(get_config("FILE_EXT"))

PAGES = ["Load Index", "Search"]
RADIO_MODES = ["Text description", "Upload image"]

st.set_page_config(page_title="Explainable Search", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def get_tools():
    return SearchArtWorks(), ArtIndexer(INDEX_FILE, META_FILE), ArtEmbedd(), ExplainITQuery()


searcher, indexer, embedder, explain_it_query = get_tools()


page = st.selectbox("Navigate", PAGES, index=0)


#  Pages - UI                                                            

if page == PAGES[0]:
    st.title("Load Index")
    st.caption("Ensure **{INDEX_FILE}** is present before loading.")

    if st.button("Load"):
        try:
            idx, meta = indexer.load_faiss_index()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state[SK_INDEX] = idx
        st.session_state[SK_META]  = meta
        st.success(f"Loaded {idx.ntotal} vectors. Switch to **Search**.")

elif page == PAGES[1]:
    st.title("iArt xAI Search")

    if SK_INDEX not in st.session_state or SK_META not in st.session_state:
        st.warning("Load or build an index first.")
        st.stop()

    index = st.session_state[SK_INDEX]
    meta  = st.session_state[SK_META]

    mode  = st.radio("Query type", RADIO_MODES, horizontal=True)
    top_k = st.slider("Top-K", 1, 15, 5)

    query = None
    
    # state for explanations
    show_explanation = st.checkbox("Display explanation for top result", value=False)
    
    if show_explanation:
        with st.expander("Explanation settings"):
            if mode == RADIO_MODES[0]:
                top_n_tokens = st.slider(
                    "Max keywords to display", min_value=1, max_value=10, value=5,
                    help="to display top N tokens sorted by absolute impact"
                )
            elif mode == RADIO_MODES[1]:
                n_grid = st.select_slider(
                    "Patch grid size", options=[4, 5, 6, 7, 8], value=5,
                    help="e.g. 4 by 4 grid"
                )

    if mode == RADIO_MODES[0]:
        query_text = st.text_input("Query")
        if query_text.strip():
            query = query_text.strip()
    else:
        uploaded = st.file_uploader("Upload image", type=FILE_EXT)
        if uploaded:
            query = Image.open(uploaded).convert("RGB")
            st.image(query, caption="Query image", width=300)


    if query is not None and st.button("Search"):
        with st.spinner("Searching…"):
            _, results = indexer.perform_similarity_search( query, embedder.get_embedding,
                            index, meta, top_k=top_k  )

        if not results:
            st.warning("No results.")
            st.stop()

        st.subheader(f"Top {len(results)} results")
        cols = st.columns(min(5, len(results)))
        
        for i, r in enumerate(results):
            with cols[i % len(cols)]:
                st.image(r.get("image_url"), width="stretch")
                st.caption(f"**{r.get('title')}** | *{r.get('artist')}*")

                st.progress(  
                    int(r.get("similarity")),
                    text=f"Similarity: {r.get('similarity'):.1f}%  (cosine: {r.get('cosine_score'):.3f})"
                )

                depicts = r.get('depicts', '')
                if depicts:
                    st.caption(f"**Depicts**: {depicts[:100]}")
                
                qid = r.get('qid', '')
                if qid:
                    st.caption(f"**QID**: {qid}")  
        

        if results and show_explanation:
            #for top #1 results
            topn = results[0:2]
            
            for i, top in enumerate(topn):
                st.divider()
                st.subheader(f"Explanation — for #{i+1}, from above result")
                st.caption(
                    f"**{top.get('title')}** by *{top.get('artist').title()}*  |  "
                    f"Similarity: {top.get('similarity'):.1f}%"
                )

                if isinstance(query, str):
                    st.markdown("**Extracted Keyword importance**")
                    st.caption(
                        "Score change when each word is removed from the query sentence. "
                        "**Higher** values mean the word was more important to securing this match."
                    )
                    
                    with st.spinner(f"Computing token importances…"):
                        token_imps = explain_it_query.explain_text_query( query,
                                top.get("image_url"),
                                embedder.get_embedding, top_n=top_n_tokens, )

                    if token_imps:
                        col1, col2 = st.columns(2)
                        with col1:
                            
                            df_tok = pd.DataFrame(token_imps)
                            
                            
                            df_tok["importance_ui"] = df_tok["importance"] * 50.0
                            
                            
                            df_tok = df_tok.set_index("token_idx")
                            df_tok = df_tok[["importance_ui"]]
                            df_tok.columns = ["Importance (% points drop)"]
                            
                            
                            df_tok.index = [idx.split("_", 1)[1] for idx in df_tok.index]
                            
                            st.bar_chart(df_tok)

                        with col2:
                            st.image(top.get("image_url"), caption=f"Top #{i+1} Result Match", width="stretch")
                    else:
                        st.info("Could not compute token importances for this result.")

                else:
                    st.markdown("**Region importance heatmap**")
                    st.caption("**Red** = Region assisted the match; **Blue** = Region which the system ignored.")

                    with st.spinner("Computing result image sensitivity..."):
                        result_heatmap, r_patch_metrics = explain_it_query.explain_image_query(
                            query, top.get("image_url"), embedder.get_embedding, grid=n_grid       )


                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(query, caption="Original Query", width="stretch")

                    with col2:
                        st.image(result_heatmap, caption=f"Top #{i+1} Result Region Importance Map", width="stretch")
                        if r_patch_metrics:
                            st.caption( f"**Region which drove the match, Highest(Red):** {r_patch_metrics.get('highest', {}).get('importance')*50.0:.2f} % points at "
                                f"Row {r_patch_metrics.get('highest', {}).get('row')}, Col {r_patch_metrics.get('highest', {}).get('col')}"   )
                            st.caption(
                                f"**Region which the system ignored, Lowest(Blue):** {r_patch_metrics.get('lowest', {}).get('importance')*50.0:.2f} % points at "
                                f"Row {r_patch_metrics.get('lowest', {}).get('row')}, Col {r_patch_metrics.get('lowest', {}).get('col')}"    )

                    with col3:
                        st.image(top.get("image_url"), caption=f"Original Top #{i+1} Result Match", width="stretch")

