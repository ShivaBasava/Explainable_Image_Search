"""

Expects a index from Phase-0 demo_data/, where an inital index to be built,  
Phase-1: -"Explainable-Multi-modal Search & Retrieval Pipeline"

Search app (WebApp 1).

"""
import io
import base64
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
from feature_visualization.ICC_P_P import ICCVisualizer
from feature_visualization.Image_color_histogram import HistogramVisualizer
from feature_visualization.UMAP import UMAPVisualizer
from XAI_Methods.GRAD_CAM import GradCAMExplainer
from XAI_Methods.Integrated_Gradients import IntegratedGradientsExplainer
from XAI_Methods.Attention_Maps import AttentionMapsExplainer

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
LOGO_FILE = "demo_data/XAI_Search_logo.png"

st.set_page_config(page_title="Explainable Search", layout="wide")


@st.cache_resource(show_spinner="Loading page....")
def get_tools():
    embedd = ArtEmbedd()
    return (SearchArtWorks(), ArtIndexer(INDEX_FILE, META_FILE), embedd, ExplainITQuery(),
            ICCVisualizer(), HistogramVisualizer(), GradCAMExplainer(embedd),
            IntegratedGradientsExplainer(embedd), AttentionMapsExplainer(embedd))

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


(searcher, indexer, embedder, explain_it_query, icc_visualizer, histogram_visualizer,
 gradcam_explainer, integrated_gradients_explainer, attention_maps_explainer) = get_tools()


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
umap_visualizer = UMAPVisualizer(embedder, index, meta)

@st.cache_data(show_spinner=False)
def _load_logo_data_uri(path, max_width=160):
    """Downscale the logo once per process and return it as a base64 data URI,
    so it can be embedded inline (as a clickable <a><img></a>) without
    re-reading/re-encoding the full-size file on every rerun."""
    try:
        logo_img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    if logo_img.width > max_width:
        ratio = max_width / logo_img.width
        logo_img = logo_img.resize((max_width, int(logo_img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    logo_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


EXAMPLE_QUERIES = ["Portrait of a smiling man", "A castle near the river"]


def _run_search(query_value, mode_value, top_k_value):
    """Embed + search, storing results into session state. Shared by the
    sidebar Search button and the landing-page example chips."""
    with st.spinner("Searching…"):
        _, results = indexer.perform_similarity_search(
            query_value, embedder.get_embedding, index, meta, top_k=top_k_value )
    st.session_state["search_query"] = query_value
    st.session_state["search_results"] = results
    st.session_state["search_mode"] = mode_value


def _select_example_query(example_text, top_k_value):
    """Button on_click callback: runs BEFORE the script reruns, so it's safe
    to set the sidebar widgets' session_state here (setting it afterwards,
    once those widgets have already been instantiated in a run, raises)."""
    st.session_state["query_mode_radio"] = RADIO_MODES[0]
    st.session_state["query_text_input"] = example_text
    _run_search(example_text, RADIO_MODES[0], top_k_value)


st.markdown(
    "<style> h1 { font-size: 3.5rem !important; font-weight: 800; } </style>",
    unsafe_allow_html=True,
)

_logo_uri = _load_logo_data_uri(LOGO_FILE)
if _logo_uri:
    st.markdown(
        f"""
        <a href="/" target="_self" style="text-decoration:none; color:inherit;">
            <div style="display:flex; align-items:center; justify-content:flex-start;
                        gap:0.75rem; cursor:pointer;">
                <img src="{_logo_uri}" alt="XAI Search logo"
                     style="height:64px; transition: transform 0.15s ease;"
                     onmouseover="this.style.transform='scale(1.06)'"
                     onmouseout="this.style.transform='scale(1)'">
                <h1 style="margin:0; font-size:3.5rem; font-weight:800;">iArt xAI Search</h1>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("iArt xAI Search")

query = None
top_n_tokens = 5

SIDEBAR_STYLE = """
<style>
/* -- Search heading -------------------------------------------------- */
[data-testid="stSidebar"] h2 {
    font-size: 2.75rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}

/* -- Consistent label / field spacing --------------------------------- */
[data-testid="stSidebar"] label {
    font-weight: 500;
}
[data-testid="stSidebar"] div[data-testid="stRadio"],
[data-testid="stSidebar"] div[data-testid="stSlider"],
[data-testid="stSidebar"] div[data-testid="stTextInput"],
[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
    margin-bottom: 1rem;
}

/* -- Radio buttons: bigger clickable area + accent color --------------- */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.6rem;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-size: 1.05rem;
    padding: 0.55rem 1rem;
    border-radius: 10px;
    border: 1.5px solid rgba(128, 128, 128, 0.35);
    cursor: pointer;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    border-color: var(--primary-color, #ff4b4b);
    background-color: color-mix(in srgb, var(--primary-color, #ff4b4b) 10%, transparent);
}
[data-testid="stSidebar"] div[role="radiogroup"] label [data-baseweb="radio"] > div:first-child {
    width: 1.2rem;
    height: 1.2rem;
}

/* -- Search button: filled accent color, slightly taller --------------- */
[data-testid="stSidebar"] div[data-testid="stButton"] button {
    height: 3rem;
    font-size: 1.05rem;
    font-weight: 600;
    border-radius: 10px;
}

/* -- Sliders: thicker track, larger thumb ------------------------------ */
[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2) {
    height: 8px;
}
[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"] {
    width: 22px;
    height: 22px;
}
</style>
"""

with st.sidebar:
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)
    st.header("Search")

    mode = st.radio("Query type", RADIO_MODES, horizontal=True, key="query_mode_radio")

    top_k_col, top_k_val_col = st.columns([5, 1])
    with top_k_col:
        top_k = st.slider("Top-K", 1, 100, 12, label_visibility="visible")
    with top_k_val_col:
        st.markdown(
            f"<div style='text-align:right; padding-top:2.3rem; font-weight:600;'>{top_k}</div>",
            unsafe_allow_html=True,
        )

    # Query input
    if mode == RADIO_MODES[0]:
        query_text = st.text_input("Search Query", key="query_text_input")
        if query_text.strip():
            query = query_text.strip()
        with st.expander("Keyword explanation settings", expanded=True):
            kw_col, kw_val_col = st.columns([5, 1])
            with kw_col:
                top_n_tokens = st.slider("Max keywords", 1, 10, 5)
            with kw_val_col:
                st.markdown(
                    f"<div style='text-align:right; padding-top:2.3rem; font-weight:600;'>{top_n_tokens}</div>",
                    unsafe_allow_html=True,
                )
    else:
        uploaded = st.file_uploader("Upload image", type=FILE_EXT)
        if uploaded:
            query = Image.open(uploaded).convert("RGB")
            st.image(query, caption="Query image", width='stretch')

    search_clicked = st.button("Search", type="primary", width='stretch')


if query is not None and search_clicked:
    _run_search(query, mode, top_k)

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

    all_urls = [r.get("image_url") for r in results if r.get("image_url")]
    image_cache = get_images_batch(all_urls)

    # Embedding Map (UMAP) — shown first; the map opens when the expander is opened.
    with st.expander("Embedding Map (UMAP)", expanded=False):
        umap_visualizer.render(query, results, image_cache)

    # Concept-level explanation from metadata ( displayed for both query types)
    with st.spinner("Computing concept alignments…"):
        concept_explainer.render(query=query, results=results, top_n=5)
    st.divider()

    # Results grid
    st.subheader(f"Top {len(results)} results")
    cols = st.columns(min(5, len(results)))

    def _show_detailed_explanation(title, result_item, result_img, idx):
        """Detailed per-result explanation, opened as a wide dialog so charts
        aren't squeezed into the narrow results-grid column."""

        @st.dialog(f"Detailed Explanation — {title}", width="large")
        def _render_dialog():
            with st.expander("Semantic Concept similarity for this result", expanded=True):
                st.caption("Shows which semantic concepts are strongly present in both the Query and this artwork.")
                with st.spinner("Computing concept overlap…"):
                    concept_explainer.render_single_result(query, result_item, n=5)

            st.divider()
            explanation_label = "Keyword Importance" if isinstance(query, str) else "Region Importance Heatmap"
            with st.expander(explanation_label, expanded=True):
                if isinstance(query, str):

                    # Text query - token importance
                    st.markdown("**Keyword importance (without stopwords)**")
                    st.caption("The chart displays the score drop when each word is removed from the Query.")
                    with st.spinner("Computing keyword importances…"):
                        token_imps = explain_it_query.explain_text_query( query, result_item.get("image_url"),
                                            embedder.get_embedding, top_n=top_n_tokens )

                    if token_imps:
                        tok_col1, tok_col2 = st.columns(2)
                        with tok_col1:

                            df_tok = pd.DataFrame(token_imps)

                            df_tok["importance_ui"] = df_tok["importance"] * 100.0

                            df_tok = df_tok.set_index("token_idx")
                            df_tok = df_tok[["importance_ui"]]
                            df_tok.columns = ["Importance (% points drop)"]

                            df_tok.index = [t.split("_", 1)[1] for t in df_tok.index]

                            st.bar_chart(df_tok, horizontal=True, y="Importance (% points drop)")

                        with tok_col2:
                            st.image(result_img, caption="Result Match", width="stretch")
                    else:
                        st.info("Could not compute token importances for this result.")

                else:

                    grid_size = st.slider( "Patch grid size for heatmap",
                        min_value=5, max_value=8, value=6, step=1,
                        help="Larger grid gives finer resolution but slower computation.",
                        key=f"grid_size_{idx}" )
                    # Image query- patch occlusion sensitivity
                    st.markdown("**Region importance heatmap**")
                    st.caption("The heatmap displays regions that most influenced the match; "
                                    "changing them may reduce the similarity score.")
                    st.caption("Red = region that most helped the match; Blue = region that was ignored.")
                    with st.spinner("Computing region importance…"):
                        result_heatmap, r_patch_metrics = explain_it_query.explain_image_query( query, result_item.get("image_url"),
                                                            embedder.get_embedding, grid=grid_size  )

                    heat_col1, heat_col2, heat_col3 = st.columns(3)
                    with heat_col1:
                        st.image(query, caption="Original Query", width="stretch")
                    with heat_col2:
                        st.image(result_heatmap, caption="Region importance", width="stretch")
                        if r_patch_metrics:

                            st.caption( f"**Highest(Red) region:** {r_patch_metrics.get('highest', {}).get('importance')*100.0:.2f} % points at "
                                    f"Row {r_patch_metrics.get('highest', {}).get('row')}, Col {r_patch_metrics.get('highest', {}).get('col')}"   )
                            st.caption(
                                f"**Lowest(Blue) region:** {r_patch_metrics.get('lowest', {}).get('importance')*100.0:.2f} % points at "
                                f"Row {r_patch_metrics.get('lowest', {}).get('row')}, Col {r_patch_metrics.get('lowest', {}).get('col')}"    )

                    with heat_col3:
                        st.image(result_img, caption="Original Retrieved result", width="stretch")

        _render_dialog()

    for i, r in enumerate(results):
        with cols[i % len(cols)]:   
            img_url = r.get("image_url")
            img_obj = None
            if img_url:

                img_obj = image_cache.get(r.get("image_url"))
                if img_obj is not None:
                    st.image(img_obj, width='stretch')
                else:
                    st.warning("Failed to download artwork.")
            else:
                st.warning("Missing image URL.")
            st.caption(f"**{r.get('title')}** | *{r.get('artist')}*")

            active_panel = st.segmented_control(
                "Details", ["Meta Data", "Image Composition", "XAI Methods"],
                selection_mode="single", default=None, key=f"panel_{i}",
                label_visibility="collapsed", width='stretch',
            )

            if img_obj is not None and st.button("Detailed Explanation", key=f"detail_btn_{i}", width='stretch'):
                _show_detailed_explanation(r.get("title", "Untitled"), r, img_obj, i)

            if active_panel == "Meta Data":
                st.progress(  int(r.get("similarity")),
                    text=f"Similarity: {r.get('similarity'):.1f}%"#  (cosine: {r.get('cosine_score'):.3f})"
                )
                st.caption(f"**Title**: {r.get('title', 'Untitled')}")
                st.caption(f"**Artist**: {r.get('artist', 'Unknown')}")

                depicts = r.get('depicts', '')
                if depicts:
                    st.caption(f"**Depicts**: {depicts}")

                description = r.get('description', '')
                if description:
                    st.caption(f"**Description**: {description}")

                qid = r.get('qid', '')
                if qid:
                    st.caption(f"**QID**: {qid}")

                wikidata_url = r.get('wikidata_url', '')
                if wikidata_url:
                    st.caption(f"**Wikidata**: {wikidata_url}")

                st.caption(f"**Cosine score**: {r.get('cosine_score', 0):.4f}")

            elif active_panel == "Image Composition":
                if img_obj is not None:
                    icc_col, hist_col = st.columns(2)
                    with icc_col:
                        if st.button("Pose Composition (ICC++)", key=f"icc_btn_{i}", width='stretch'):
                            icc_visualizer.show_dialog(r.get("title", "Untitled"), img_obj, query)
                    with hist_col:
                        if st.button("Color Histogram", key=f"hist_btn_{i}", width='stretch'):
                            histogram_visualizer.show_dialog(r.get("title", "Untitled"), img_obj)
                else:
                    st.info("Artwork image unavailable.")

            elif active_panel == "XAI Methods":
                if img_obj is not None:
                    att_col, gradcam_col, ig_col = st.columns(3)
                    with att_col:
                        if st.button("Attention Map", key=f"attn_btn_{i}", width='stretch'):
                            attention_maps_explainer.show_dialog(r.get("title", "Untitled"), img_obj, query)
                    with gradcam_col:
                        if st.button("Grad-CAM (Query Relevance)", key=f"gradcam_btn_{i}", width='stretch'):
                            gradcam_explainer.show_dialog(r.get("title", "Untitled"), img_obj, query)
                    with ig_col:
                        if st.button("Integrated Gradients", key=f"ig_btn_{i}", width='stretch'):
                            integrated_gradients_explainer.show_dialog(r.get("title", "Untitled"), img_obj, query)
                else:
                    st.info("Artwork image unavailable.")

else:
    # Landing state — no search performed yet. Centered, interactive hero.
    st.markdown(
        """
        <style>
        .st-key-landing_hero [data-testid="stButton"] button {
            border-radius: 999px;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }
        .st-key-landing_hero [data-testid="stButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
            border-color: var(--primary-color, #ff4b4b);
            color: var(--primary-color, #ff4b4b);
        }
        .st-key-landing_hero .landing-tagline {
            text-align: center;
            font-size: 2rem;
            opacity: 0.85;
            margin-bottom: 1.75rem;
        }
        .st-key-landing_hero .landing-try {
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;
            opacity: 0.7;
            margin-bottom: 0.5rem;
        }
        .st-key-landing_hero .landing-spacer {
            height: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="landing_hero"):
        st.markdown("<div class='landing-spacer'></div>", unsafe_allow_html=True)
        _, hero_col, _ = st.columns([1, 2, 1])
        with hero_col:
            st.markdown(
                "<div class='landing-tagline'>Search using text or upload an image.</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div class='landing-try'>Try:</div>", unsafe_allow_html=True)
            example_cols = st.columns(len(EXAMPLE_QUERIES))
            for example_col, example in zip(example_cols, EXAMPLE_QUERIES):
                with example_col:
                    st.button(
                        example, key=f"example_{example}", width='stretch',
                        on_click=_select_example_query, args=(example, top_k),
                    )


