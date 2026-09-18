"""
ALGORITHM: Attention maps for the SigLIP2-NaFlex vision encoder

Query-independent — this shows how the vision transformer *reads* an artwork
on its own, the same for every search (unlike Grad-CAM / Integrated Gradients,
which explain one specific query↔result match).

Two maps per image:

  (1) Attention rollout  (Abnar & Zuidema, 2020)
        per encoder layer:  A_l = mean over heads of softmax(Q Kᵀ / √d)
        add the residual connection:  A_l ← (A_l + I) / row-sum
        propagate through the first D layers:  R = A_D @ … @ A_1
        per-patch score = mean attention each patch *receives* = R.mean(rows)
      A "rollout depth" slider sets D, so sliding from 1 → L shows the
      attention concentrating layer by layer.

  (2) Pooling-head attention
        the attention-pooling head owns a learnable "probe" query that attends
        over every patch token; its attention weights are exactly how the patch
        features get pooled into the single image embedding the FAISS index
        stores — i.e. "what the model kept for retrieval".

SDPA does not expose attention weights, so (1) is recomputed directly from
each layer's own q_proj / k_proj via forward-pre-hooks — no reliance on
`output_attentions`, and no second copy of the model. (2) is read from a
forward hook on the pooling head's `torch.nn.MultiheadAttention`.

Both maps are stripped of NaFlex padding, reshaped to the true (h, w) patch
grid, upsampled to the image and overlaid (JET colormap).
"""

import hashlib

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


@st.cache_data(show_spinner=False, max_entries=64)
def _collect_attention(_embedder, _image, image_key, num_patches):
    """Run the vision tower once and return everything the maps need, as numpy:

        per_layer : [L, N, N] float32   mean-over-heads self-attention per layer
        probe     : [N] float32 or None  pooling-head probe attention over patches
        keep      : [N] bool             real (non-padded) patch positions
        (h, w)    : true NaFlex patch grid

    `_embedder` / `_image` are excluded from hashing (leading underscore);
    `image_key` + `num_patches` are the cache key.
    """
    model = _embedder.model
    # `model.vision_model` is either the Siglip2VisionTransformer itself or a
    # Siglip2VisionModel wrapper around it (varies by transformers version);
    # unwrap to the transformer so the encoder layers / pooling head and the
    # forward signature (`attention_mask=`) are always the same.
    vision = getattr(model.vision_model, "vision_model", model.vision_model)
    layers = vision.encoder.layers
    processor = _embedder.processor

    inputs = processor(
        images=_image.convert("RGB"), return_tensors="pt", max_num_patches=num_patches
    ).to(_embedder.device)
    keep_t = inputs["pixel_attention_mask"][0].bool()
    h, w = (int(v) for v in inputs["spatial_shapes"][0].tolist())

    # -- capture each layer's (already layer-norm'd) attention input -----
    layer_input = {}

    def _pre_hook_factory(idx):
        def _hook(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            layer_input[idx] = hs.detach()
        return _hook

    head_weights = {}

    def _head_hook(module, args, output):
        # torch.nn.MultiheadAttention -> (attn_output, attn_weights[b, tgt=1, src=N])
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            head_weights["w"] = output[1].detach()

    handles = [
        layers[i].self_attn.register_forward_pre_hook(_pre_hook_factory(i), with_kwargs=True)
        for i in range(len(layers))
    ]
    handles.append(vision.head.attention.register_forward_hook(_head_hook))
    try:
        with torch.no_grad():
            try:
                vision(
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["pixel_attention_mask"],
                    spatial_shapes=inputs["spatial_shapes"],
                )
            except TypeError:
                # older/wrapped signature uses `pixel_attention_mask=`
                vision(
                    pixel_values=inputs["pixel_values"],
                    pixel_attention_mask=inputs["pixel_attention_mask"],
                    spatial_shapes=inputs["spatial_shapes"],
                )
    finally:
        for handle in handles:
            handle.remove()

    # -- recompute per-layer attention from q_proj / k_proj -------------
    neg_inf = torch.finfo(torch.float32).min
    per_layer = []
    with torch.no_grad():
        for i, layer in enumerate(layers):
            hs = layer_input[i].float()
            attn = layer.self_attn
            bsz, n_tok, _ = hs.shape
            n_heads, head_dim = attn.num_heads, attn.head_dim

            q = attn.q_proj(hs).view(bsz, n_tok, n_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hs).view(bsz, n_tok, n_heads, head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) * attn.scale          # [B, H, N, N]
            scores = scores.masked_fill(~keep_t.view(1, 1, 1, n_tok), neg_inf)
            a = scores.softmax(dim=-1).mean(dim=1)[0]                 # [N, N]
            per_layer.append(torch.nan_to_num(a, nan=0.0).cpu().numpy().astype(np.float32))

    per_layer = np.stack(per_layer, axis=0)
    keep = keep_t.cpu().numpy()
    probe = (
        head_weights["w"][0, 0].cpu().numpy().astype(np.float32)
        if "w" in head_weights else None
    )
    return per_layer, probe, keep, (h, w)


class AttentionMapsExplainer:
    """Attention-rollout + pooling-head attention maps for the SigLIP2-NaFlex
    vision encoder. Reuses the ArtEmbedd model so the maps run on the exact
    weights the search index was built with."""

    HEATMAP_ALPHA = 0.45
    NUM_PATCHES = 256   # match the resolution ArtEmbedd used to build the index

    def __init__(self, art_embedder):
        self.art_embedder = art_embedder

    # ------------------------------------------------------------------
    @staticmethod
    def _image_key(image: Image.Image) -> str:
        return hashlib.md5(image.convert("RGB").resize((64, 64)).tobytes()).hexdigest()

    @staticmethod
    def _rollout(per_layer: np.ndarray, keep: np.ndarray, depth: int) -> np.ndarray:
        """Residual-corrected attention rollout through the first `depth` layers;
        returns the mean attention received by each real patch."""
        n = per_layer.shape[1]
        eye = np.eye(n, dtype=np.float32)
        rolled = eye.copy()
        for a in per_layer[:depth]:
            a = a + eye
            a = a / a.sum(axis=-1, keepdims=True)
            rolled = a @ rolled
        return rolled[keep][:, keep].mean(axis=0)

    def _overlay_heatmap(self, image: Image.Image, cam: np.ndarray) -> Image.Image:
        rgb_image = image.convert("RGB")
        w, h = rgb_image.size
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        base = np.array(rgb_image, dtype=np.uint8)
        blended = cv2.addWeighted(heatmap_rgb, self.HEATMAP_ALPHA, base, 1 - self.HEATMAP_ALPHA, 0)
        return Image.fromarray(blended)

    def _to_overlay(
        self, image: Image.Image, flat: np.ndarray, grid_hw, suppress_border: bool = True
    ) -> Image.Image:
        """Reshape a per-patch vector to the patch grid, robust-normalise to
        [0, 1] and overlay it on the image.

        SigLIP/ViT models dump a lot of attention onto a few high-norm "sink"
        patches around the image border; when `suppress_border` is set those
        edge patches are capped at the interior 75th percentile so they stop
        dominating the colour scale (no hard zeroing -> no dark frame)."""
        h, w = grid_hw
        grid = np.asarray(flat, dtype=np.float32)[: h * w].reshape(h, w).copy()

        if suppress_border and h > 2 and w > 2:
            edge = np.ones((h, w), dtype=bool)
            edge[1:-1, 1:-1] = False
            cap = float(np.percentile(grid[~edge], 75))
            grid[edge] = np.minimum(grid[edge], cap)

        hi = float(np.percentile(grid, 99))
        grid = np.clip(grid / hi, 0.0, 1.0) if hi > 1e-8 else np.zeros_like(grid)
        return self._overlay_heatmap(image, grid)

    # ------------------------------------------------------------------
    # Streamlit dialog
    # ------------------------------------------------------------------
    def show_dialog(self, title: str, image: Image.Image, query=None):
        """Open the attention-maps dialog for one result image. `query` is
        accepted for a consistent call signature but not used — attention maps
        are query-independent."""

        @st.dialog(f"Attention Maps — {title}", width="medium")
        def _render_dialog():
            with st.spinner("Running the vision encoder…"):
                try:
                    per_layer, probe, keep, grid_hw = _collect_attention(
                        self.art_embedder, image, self._image_key(image), self.NUM_PATCHES
                    )
                except Exception as e:
                    st.error(f"Could not compute attention maps: {e}")
                    return

            n_layers = int(per_layer.shape[0])
            ctrl_col, chk_col = st.columns([3, 2])
            with ctrl_col:
                depth = st.slider(
                    "Rollout through layer", 1, n_layers, n_layers,
                    help="How many encoder layers to accumulate. Slide from 1 upward "
                         "to watch the attention concentrate layer by layer.",
                )
            with chk_col:
                suppress_border = st.checkbox(
                    "De-emphasise border sinks", value=True,
                    help="ViT models pour attention into a few high-norm patches at the "
                         "image edge. Uncheck to see that raw behaviour.",
                )
            rollout = self._rollout(per_layer, keep, depth)

            roll_col, pool_col = st.columns(2)
            with roll_col:
                st.markdown("**Attention rollout**")
                st.image(self._to_overlay(image, rollout, grid_hw, suppress_border), width="stretch")
                st.caption(
                    f"Encoder self-attention accumulated through layer {depth}/{n_layers}. "
                    "Red = the patches the network's attention keeps returning to."
                )
            with pool_col:
                st.markdown("**Pooling-head attention**")
                if probe is not None:
                    st.image(self._to_overlay(image, probe, grid_hw, suppress_border), width="stretch")
                    st.caption(
                        "Where the attention-pooling head looks when it compresses the "
                        "patch features into the image embedding used for search."
                    )
                else:
                    st.info("Pooling-head attention is unavailable for this model build.")

            st.caption(
                f"Patch grid {grid_hw[0]}×{grid_hw[1]}. Query-independent — this is how the "
                "SigLIP2 vision encoder reads the image itself, identically for every search."
            )

        _render_dialog()
