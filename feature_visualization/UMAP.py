"""
UMAP embedding-space view for a search result set  (WebApp-1, feature_visualization/).

Projects the SigLIP2 query vector together with the retrieved artworks — over the
full FAISS index as faint background context — down to 2D with UMAP, so you can
see *where* the results land in the embedding space and how tightly they cluster
around the query.

Rendered inline inside an expander (UMAPVisualizer.render); the map is built
when the expander is opened. Two UMAP hyper-parameters are exposed as sliders:
    - n_neighbors : balances local vs. global structure
                    (low  -> local neighbourhoods, high -> overall layout)
    - min_dist    : how tightly UMAP is allowed to pack points together
                    (low  -> dense clumps, high -> more even spread)

This is a read-only visualisation of the exact vectors the search ran on; it
does not change the results in any way.
"""

import base64
import hashlib
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:
    import faiss
except Exception:  # faiss ships with the app; guard only so import never hard-fails
    faiss = None


# ----------------------------------------------------------------------
# Cached heavy steps
# (module-level functions cache more predictably than bound methods)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _reconstruct_corpus(_index, _meta, index_signature):
    """Pull every stored (already L2-normalised) vector out of the FAISS index
    together with its metadata item.

    Returns (vectors [N, D] float32, items list) or (None, []) when the index
    can't be read. `index_signature` (index.ntotal) is the real cache key —
    `_index` / `_meta` are excluded from hashing by the leading underscore.
    """
    if _index is None or faiss is None:
        return None, []
    try:
        flat = faiss.downcast_index(_index.index)  # unwrap IndexIDMap -> IndexFlatIP
        n = int(flat.ntotal)
        if n == 0:
            return None, []
        vecs = np.asarray(flat.reconstruct_n(0, n), dtype=np.float32)
        ids = faiss.vector_to_array(_index.id_map).astype(np.int64)
    except Exception:
        return None, []

    items_by_id = {int(it.get("id")): it for it in _meta.get("items", [])}
    items = [items_by_id.get(int(i), {"id": int(i)}) for i in ids]
    return vecs, items


@st.cache_data(show_spinner=False)
def _embed_query(_embedder, _query, query_key):
    """L2-normalised embedding of the text/image query. `query_key` carries the
    identity used for caching (see UMAPVisualizer._query_key)."""
    q = _query.strip().lower() if isinstance(_query, str) else _query
    vec = np.asarray(_embedder.get_embedding(q), dtype=np.float32).reshape(-1)
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


@st.cache_data(show_spinner=False)
def _embed_images(_embedder, _images, images_key):
    """L2-normalised embeddings for a list of result images. Entries that are
    None (download failed, or vector already available from the index) stay
    None. `images_key` — a tuple of (id, url) — is the cache key."""
    out = []
    for img in _images:
        if img is None:
            out.append(None)
            continue
        vec = np.asarray(_embedder.get_embedding(img), dtype=np.float32).reshape(-1)
        out.append(vec / max(float(np.linalg.norm(vec)), 1e-8))
    return out


@st.cache_data(show_spinner=False)
def _run_umap(points, n_neighbors, min_dist, seed):
    """Fit UMAP on the stacked point cloud and return the 2D coordinates.
    Cached on (points, n_neighbors, min_dist) so moving a slider only re-fits
    UMAP — the embeddings above are reused."""
    import umap

    n = len(points)
    k = int(np.clip(n_neighbors, 2, max(2, n - 1)))
    reducer = umap.UMAP(
        n_neighbors=k,
        min_dist=float(min_dist),
        n_components=2,
        metric="cosine",
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(np.asarray(points, dtype=np.float32)))


class UMAPVisualizer:
    """2D UMAP projection of the query + retrieved artworks over the index."""

    # slider ranges / defaults
    N_NEIGHBORS_MIN, N_NEIGHBORS_MAX, N_NEIGHBORS_DEFAULT = 2, 50, 4
    MIN_DIST_MIN, MIN_DIST_MAX, MIN_DIST_DEFAULT = 0.0, 0.99, 0.1

    MAX_BACKGROUND = 1500  # cap corpus points fed to UMAP (keeps it responsive)
    RANDOM_STATE = 42

    def __init__(self, embedder, index=None, meta=None):
        self.embedder = embedder
        self.index = index
        self.meta = meta or {}

    # ------------------------------------------------------------------
    # caching identity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _query_key(query):
        if isinstance(query, str):
            return f"text::{query.strip().lower()}"
        thumb = query.convert("RGB").resize((64, 64))
        return "image::" + hashlib.md5(thumb.tobytes()).hexdigest()

    # ------------------------------------------------------------------
    # assemble the point cloud
    # ------------------------------------------------------------------
    def _collect_points(self, query, results, image_map):
        """Gather query / result / background vectors + parallel metadata.
        Returns a dict, or None if not even the results could be embedded."""
        sig = None if self.index is None else int(self.index.ntotal)
        corpus_vecs, corpus_items = _reconstruct_corpus(self.index, self.meta, sig)

        q_vec = _embed_query(self.embedder, query, self._query_key(query))

        pos_by_id = {}
        if corpus_vecs is not None:
            pos_by_id = {int(it.get("id")): p for p, it in enumerate(corpus_items)
                         if it.get("id") is not None}

        # Re-embed a result image only when its stored index vector is missing.
        urls = [r.get("image_url") for r in results]
        to_embed = [
            (image_map.get(u) if (corpus_vecs is None or r.get("id") not in pos_by_id) else None)
            for r, u in zip(results, urls)
        ]
        embedded = _embed_images(
            self.embedder, to_embed,
            tuple((r.get("id"), u) for r, u in zip(results, urls)),
        )

        res_vecs, res_meta, res_imgs = [], [], []
        for r, emb in zip(results, embedded):
            rid = r.get("id")
            if corpus_vecs is not None and rid in pos_by_id:
                res_vecs.append(corpus_vecs[pos_by_id[rid]])
            elif emb is not None:
                res_vecs.append(emb)
            else:
                continue
            res_meta.append(r)
            res_imgs.append(image_map.get(r.get("image_url")))

        if not res_vecs:
            return None

        result_ids = {r.get("id") for r in res_meta}

        # Background = the rest of the collection (results excluded so a point
        # is never drawn twice).
        bg_vecs, bg_meta = [], []
        if corpus_vecs is not None:
            for vec, it in zip(corpus_vecs, corpus_items):
                if it.get("id") in result_ids:
                    continue
                bg_vecs.append(vec)
                bg_meta.append(it)
            if len(bg_vecs) > self.MAX_BACKGROUND:
                rng = np.random.default_rng(self.RANDOM_STATE)
                keep = sorted(rng.choice(len(bg_vecs), self.MAX_BACKGROUND, replace=False))
                bg_vecs = [bg_vecs[i] for i in keep]
                bg_meta = [bg_meta[i] for i in keep]

        return {
            "query_vec": q_vec,
            "res_vecs": np.asarray(res_vecs, dtype=np.float32),
            "res_meta": res_meta,
            "res_imgs": res_imgs,
            "bg_vecs": np.asarray(bg_vecs, dtype=np.float32) if bg_vecs else None,
            "bg_meta": bg_meta,
        }

    # ------------------------------------------------------------------
    # projection + figure
    # ------------------------------------------------------------------
    def _project(self, cloud, n_neighbors, min_dist):
        parts = [cloud["query_vec"][None, :], cloud["res_vecs"]]
        if cloud["bg_vecs"] is not None:
            parts.append(cloud["bg_vecs"])
        stacked = np.vstack(parts).astype(np.float32)

        coords = _run_umap(stacked, int(n_neighbors), float(min_dist), self.RANDOM_STATE)

        n_res = len(cloud["res_vecs"])
        return coords[0], coords[1:1 + n_res], coords[1 + n_res:]

    @staticmethod
    def _thumb_uri(img, size=96):
        """Centre-crop an artwork image to a square and return it as a small
        PNG data URI, for use as a Plotly layout image."""
        im = img.convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _sim_colors(sims):
        """One colour per result, sampled from YlOrRd by relative similarity —
        used for the border drawn around each thumbnail icon."""
        if not sims:
            return []
        lo, hi = min(sims), max(sims)
        rng = hi - lo
        fracs = [0.5 if rng < 1e-9 else (s - lo) / rng for s in sims]
        try:
            from plotly.colors import sample_colorscale
            return sample_colorscale("YlOrRd", fracs)
        except Exception:
            return ["#f58518"] * len(sims)

    def _figure(self, cloud, q_xy, res_xy, bg_xy):
        fig = go.Figure()

        if len(bg_xy):
            fig.add_trace(go.Scatter(
                x=bg_xy[:, 0], y=bg_xy[:, 1], mode="markers",
                name="Rest of collection",
                marker=dict(size=5, color="rgba(150,150,150,0.35)"),
                hovertext=[it.get("title", "Untitled") for it in cloud["bg_meta"]],
                hoverinfo="text",
            ))

        sims = [float(r.get("similarity", 0.0)) for r in cloud["res_meta"]]
        res_imgs = cloud.get("res_imgs") or [None] * len(cloud["res_meta"])
        border_colors = self._sim_colors(sims)

        # Icon size + axis padding, in DATA units — layout images and shapes do
        # not expand the axis autorange, so both are derived from the point spread
        # and the ranges are set explicitly below.
        all_xy = np.vstack([p for p in (q_xy[None, :], res_xy, bg_xy) if len(p)])
        span = float(max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]))) or 1.0
        icon = span * 0.045
        pad = icon * 1.4

        # One thumbnail icon (with a similarity-coloured border) per result;
        # a coloured dot is the fallback when the image could not be downloaded.
        dot_x, dot_y, dot_c = [], [], []
        for (x, y), img, col in zip(res_xy, res_imgs, border_colors):
            if img is None:
                dot_x.append(x)
                dot_y.append(y)
                dot_c.append(col)
                continue
            fig.add_shape(
                type="rect",
                x0=x - icon * 0.62, x1=x + icon * 0.62,
                y0=y - icon * 0.62, y1=y + icon * 0.62,
                line=dict(color=col, width=3), fillcolor="white",
                layer="below",
            )
            fig.add_layout_image(dict(
                source=self._thumb_uri(img), xref="x", yref="y",
                x=x, y=y, sizex=icon, sizey=icon,
                xanchor="center", yanchor="middle",
                sizing="contain", layer="above",
            ))
        if dot_x:
            fig.add_trace(go.Scatter(
                x=dot_x, y=dot_y, mode="markers", showlegend=False,
                marker=dict(size=13, color=dot_c, line=dict(width=1, color="#444")),
                hoverinfo="skip",
            ))

        # Invisible markers at each exact result point: carry the hover text and
        # drive the "Similarity %" colour bar.
        fig.add_trace(go.Scatter(
            x=res_xy[:, 0], y=res_xy[:, 1], mode="markers", showlegend=False,
            marker=dict(
                size=6, color=sims, colorscale="YlOrRd", opacity=0.0,
                cmin=min(sims) if sims else 0.0, cmax=max(sims) if sims else 100.0,
                showscale=True, colorbar=dict(title="Similarity %"),
            ),
            hovertext=[
                f"#{i + 1} — {r.get('title', 'Untitled')}<br>"
                f"{r.get('artist', 'Unknown')}<br>"
                f"Similarity {float(r.get('similarity', 0.0)):.1f}%"
                for i, r in enumerate(cloud["res_meta"])
            ],
            hoverinfo="text",
        ))

        # Rank labels, lifted clear of the icons.
        fig.add_trace(go.Scatter(
            x=res_xy[:, 0], y=res_xy[:, 1] + icon * 0.72, mode="text",
            text=[str(i + 1) for i in range(len(res_xy))],
            textposition="top center", textfont=dict(size=13, color="#111111"),
            showlegend=False, hoverinfo="skip",
        ))

        # Legend proxy for the result thumbnails (off-canvas point).
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="Search results",
            marker=dict(size=11, color="#f58518", symbol="square",
                        line=dict(width=2, color="#c2410c")),
        ))

        fig.add_trace(go.Scatter(
            x=[q_xy[0]], y=[q_xy[1]], mode="markers",
            name="Query",
            marker=dict(size=20, color="#111111", symbol="star",
                        line=dict(width=1, color="#ffffff")),
            hovertext=["Query"], hoverinfo="text",
        ))

        fig.update_layout(
            height=560,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None,
                       range=[all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad]),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None,
                       scaleanchor="x", scaleratio=1,
                       range=[all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad]),
        )
        return fig

    # ------------------------------------------------------------------
    # Inline render (e.g. inside an expander)
    # ------------------------------------------------------------------
    def render(self, query, results, image_map=None):
        """Render the UMAP embedding map for the current result set inline,
        with n_neighbors / min_dist sliders. Meant to be called from inside an
        expander so the map is built when the expander is opened."""
        image_map = image_map or {}

        try:
            import umap  # noqa: F401
        except Exception:
            st.error(
                "UMAP is not installed. Install it and reload:\n\n"
                "```\npip install umap-learn\n```"
            )
            return

        st.caption(
            "2D UMAP projection of the SigLIP2 vectors the search ran on. "
            "★ = your query, thumbnails = the ranked results (border shade = similarity), "
            "faint grey points = the rest of the indexed collection."
        )

        c1, c2 = st.columns(2)
        with c1:
            n_neighbors = st.slider(
                "n_neighbors", self.N_NEIGHBORS_MIN, self.N_NEIGHBORS_MAX,
                self.N_NEIGHBORS_DEFAULT,
                help="Low = preserves local neighbourhoods; "
                     "high = preserves the overall shape of the embedding space.",
            )
        with c2:
            min_dist = st.slider(
                "min_dist", self.MIN_DIST_MIN, self.MIN_DIST_MAX,
                self.MIN_DIST_DEFAULT, step=0.01,
                help="Low = points may clump tightly; high = points spread more evenly.",
            )

        with st.spinner("Collecting vectors…"):
            cloud = self._collect_points(query, results, image_map)
        if cloud is None:
            st.warning("Could not gather enough vectors to project this result set.")
            return

        total = (
            1 + len(cloud["res_vecs"])
            + (len(cloud["bg_vecs"]) if cloud["bg_vecs"] is not None else 0)
        )
        if total < 4:
            st.info("Need at least a few points for a UMAP projection — try a larger Top-K.")
            return

        with st.spinner("Running UMAP…"):
            try:
                q_xy, res_xy, bg_xy = self._project(cloud, n_neighbors, min_dist)
            except Exception as e:
                st.error(f"Could not compute the UMAP projection: {e}")
                return

        st.plotly_chart(self._figure(cloud, q_xy, res_xy, bg_xy), use_container_width=True)
        st.caption(
            "UMAP axes have no units and absolute distances are not meaningful — "
            "read this as *which points group together*, not how far apart they are."
        )
