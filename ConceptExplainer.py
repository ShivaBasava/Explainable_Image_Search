import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

DEFAULT_CONCEPTS = [ "Baroque", "Gothic", "Renaissance", "Impressionism", "Flemish", "Dutch", "Landscape", "Portrait",
    "Still Life", "Seascape", "Interior", "Canvas", "Panel", "Oil Painting", "Etching", "Woodcut",
    "Fresco", "Mountain", "Allegory", "Biblical", "Watercolor", "Castle", "Church", "Mythology" ]

#Its the fields from which we extract text for embedding and semantic concept explanation.
METADATA_FIELDS = ["title", "artist", "depicts", "description"] 

class ConceptExplainer:
    """a metadata-based(file _art.meta.json) semantic concept explanation, 
    using a the current embedding functionality of the ArtEmbedder."""

    def __init__(self, embedder, index=None, meta=None, concepts=None):
        self.embedder = embedder
        self.index = index
        self.meta = meta or {}
        self.concepts = concepts or DEFAULT_CONCEPTS

        self._concept_embs = None
        self._concept_emb_cache = {}
        self._art_activation_cache = {}

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)

    def _ensure_concepts(self):
        if self._concept_embs is not None:
            return

        embs = []
        for concept in self.concepts:
            # Handle text input for string concepts
            key = concept.strip().lower() if isinstance(concept, str) else concept
            if key not in self._concept_emb_cache:
                raw = self.embedder.get_embedding(key)
                self._concept_emb_cache[key] = self._normalize(raw)
            embs.append(self._concept_emb_cache[key])

        self._concept_embs = np.vstack(embs).astype(np.float32)

    def _extract_text_context(self, r_item: dict) -> str:
        """from metadata fields, 
        extracting a text block for embedding."""
        parts = []
        for field in METADATA_FIELDS:
            val = (r_item.get(field) or "").strip()
            if val:
                parts.append(f"{field} {val}")
        return " ".join(parts).strip().lower()

    def _bar(self, pairs, title):
        if not pairs:
            st.info("No concepts found for this view.")
            return

        labels = [p[0] for p in pairs]
        # get values as percentages
        values = [p[1] * 100.0 for p in pairs]

        fig = go.Figure(go.Bar( x=values, y=labels, orientation="h",
            text=[f"{v:.2f} %" for v in values],
            textposition="outside", marker_color="#ff7f0e",  ))

        fig.update_layout( title=title,
            xaxis_title="Concept Similarity Score", yaxis_title="Concept", 
            height=max(240, 34 * len(labels) + 80),
            margin=dict(l=20, r=30, t=50, b=20),
            showlegend=False,
            yaxis=dict(autorange="reversed") )

        st.plotly_chart(fig, use_container_width=True)

    def concept_projection(self, query):
        """compute concept projection for 
        a Query string or image,"""
        self._ensure_concepts()

        if isinstance(query, str):
            processed_query = query.strip().lower()
        else:
            processed_query = query

        q_vec = self._normalize(self.embedder.get_embedding(processed_query))
        return (self._concept_embs @ q_vec).astype(np.float32) #l2 normalized dot product, cosine similarity

    def artwork_concept_projection(self, r_item: dict):
        """Compute concept projection for a single artwork.
        place it in a cache for later."""

        self._ensure_concepts()
        qid = r_item.get("qid")

        if qid in self._art_activation_cache:
            return self._art_activation_cache[qid]

        context_text = self._extract_text_context(r_item)
        if not context_text:
            return None

        try:
            art_vec = self._normalize(self.embedder.get_embedding(context_text))
            activation = (self._concept_embs @ art_vec).astype(np.float32)
            if qid:
                self._art_activation_cache[qid] = activation
            return activation
        except Exception:
            return None

    def result_overlap(self, r_item: dict, query_act: np.ndarray, n=3):
        """result overlap of the concepts, 
        for a single search result."""
        res_act = self.artwork_concept_projection(r_item)
        if res_act is None or len(query_act) == 0:
            return []

        overlap = np.maximum(query_act, 0.0) * np.maximum(res_act, 0.0)
        idx = np.argsort(overlap)[::-1][:n]
        return [(self.concepts[i], float(overlap[i])) for i in idx if overlap[i] > 0]

    def listwise_overlap(self, results: list, query_act: np.ndarray, n=5):
        """result overlap of the concepts,for a list of current searchresults, 
        avg across the list."""
        self._ensure_concepts()
        if not results:
            return []

        accum = np.zeros(len(self.concepts), dtype=np.float32)
        count = 0

        for r in results:
            act = self.artwork_concept_projection(r)
            if act is not None:
                accum += np.maximum(query_act, 0.0) * np.maximum(act, 0.0)
                count += 1

        if count == 0:
            return []

        accum /= count
        idx = np.argsort(accum)[::-1][:n]
        return [(self.concepts[i], float(accum[i])) for i in idx if accum[i] > 0]

    def render(self, query, results, top_n=8, n_result=5, n_listwise=5):
        """ Display semantic concept Top-5 explanation for the Query & Top-N Results.
        """
        if not self.concepts:
            st.warning("No concepts loaded.")
            return

        query_act = self.concept_projection(query)
        # non-negative values, for query activations
        clamped_query_act = np.maximum(query_act, 0.0).tolist()
        top_q = sorted(zip(self.concepts, clamped_query_act), key=lambda x: x[1], reverse=True)[:top_n]

        with st.expander("Concept vocabulary used for Semantic Explanation", expanded=False):
            st.caption(f"Loaded {len(self.concepts)} concepts")
            cols = st.columns(6)
            for i, c in enumerate(self.concepts):
                cols[i % 6].write(f"- {c}")

        c1, c2 = st.columns(2)

        with c1:
            label = "Query Image" if isinstance(query, Image.Image) else "Query Text"
            with st.expander("Query Semantic Concepts", expanded=True):
                self._bar(top_q, f"{label} — Top {top_n} Concepts")

                # Extract up to 3 highest scoring text features, for display
                top_q_concepts = [f"**{p[0]}**" for p in top_q if p[1] > 0][:3]

                if top_q_concepts:
                    if len(top_q_concepts) == 3:
                        concepts_str = f"{top_q_concepts[0]}, {top_q_concepts[1]}, and {top_q_concepts[2]}"
                    elif len(top_q_concepts) == 2:
                        concepts_str = f"{top_q_concepts[0]} and {top_q_concepts[1]}"
                    else:
                        concepts_str = top_q_concepts[0]

                    st.info( f"The Query is most strongly associated with the concepts {concepts_str}, "
                        f"suggesting semantic concepts likely contributed to the retrieved results." )

        with c2:
            with st.expander("Shared Semantic Concepts Across Results", expanded=True):
                shared = self.listwise_overlap(results, query_act, n_listwise)
                self._bar(shared, f"Shared Semantic Concepts Across Top-{len(results)} Results")

                # up to 3 shared features, for display
                shared_concepts = [f"**{p[0]}**" for p in shared if p[1] > 0][:3]

                if shared_concepts:
                    if len(shared_concepts) == 3:
                        shared_str = f"{shared_concepts[0]}, {shared_concepts[1]} and {shared_concepts[2]}"
                    elif len(shared_concepts) == 2:
                        shared_str = f"{shared_concepts[0]} and {shared_concepts[1]}"
                    else:
                        shared_str = shared_concepts[0]
                        
                    st.info( f"The retrieved artworks share the concepts {shared_str}, "
                        f"indicating that shared semantic concepts likely contributed to the retrieved results." )
        
    def render_single_result(self, query, result, n=5):
        """display Top-N concept overlap for a single result.
        """
        query_act = self.concept_projection(query)
        overlap = self.result_overlap(result, query_act, n=n)
        if overlap:
            self._bar(overlap, f"Top {n} Shared Concepts — {result.get('title', 'Result')}")
        else:
            st.info("No overlapping concepts found for this result.")

