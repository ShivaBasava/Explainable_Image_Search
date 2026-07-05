"""
ALGORITHM: Integrated Gradients for SigLIP2-NaFlex
    (a) text query  -> image explanation
    (b) image query -> image explanation

INPUT:
    x        # preprocessed (NaFlex-patchified) result image tensor - the input
    query    # text string, OR a query PIL image
    model    # siglip2-base-patch16-naflex (reused from ArtEmbedd)
    steps    # e.g. 50

STEP 0: Baseline
    x' <- zeros_like(x)              # black image (or blurred x)

STEP 1: Accumulate gradients along the straight-line path x' -> x
    query_emb <- text_encoder(query) -> normalize        # (a) frozen anchor
              <- MAP_head(query image) -> normalize      # (b) frozen anchor
    total_grad <- 0
    for k = 1 .. steps:
        alpha <- k / steps
        xk <- x' + alpha . (x - x')      # interpolated image
        cos <- dot(img_emb(xk), query_emb)
        z   <- cos                       # logit_scale/logit_bias are a positive
                                          # affine transform of cos, constant at
                                          # every step, so they only rescale the
                                          # final integrated gradients by a
                                          # constant factor - removed anyway by
                                          # the STEP 3 [0, 1] normalization
        total_grad <- total_grad + dz/dxk    # backward pass

    avg_grad <- total_grad / steps

STEP 2: Integrated Gradients formula
    IG <- (x - x') (x) avg_grad        # element-wise

STEP 3: Reduce to a saliency map
    IG <- |IG|                        # absolute value
    IG <- max over RGB channels       # -> single-channel map (per patch, here)
    normalize to [0,1], upsample, overlay
"""

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class IntegratedGradientsExplainer:
    """Integrated Gradients explanation for SigLIP2-NaFlex, for both search modes:
    text query -> image explanation, and image query -> image explanation.

    Attributes input-patch importance by integrating the gradient of the match
    score along a straight-line path from a black baseline to the actual
    (preprocessed) result image, then reduces it to a per-patch saliency map.
    """

    STEPS = 50                  # integration steps along baseline -> input path
    MAX_LENGTH = 64              # matches ArtEmbedd's text tokenization length
    HEATMAP_ALPHA = 0.45         # overlay blend weight

    def __init__(self, art_embedder):
        """art_embedder: an ArtEmbedd instance. Reused so Integrated Gradients
        runs on the exact same loaded SigLIP2 weights/processor/device as the
        search index, instead of loading a second copy of the model."""
        self.art_embedder = art_embedder

    # ------------------------------------------------------------------
    # Query anchors (frozen - no gradient needed)
    # ------------------------------------------------------------------
    def _encode_text(self, text: str) -> torch.Tensor:
        """(a) Frozen, L2-normalized text embedding - the query anchor."""
        processor = self.art_embedder.processor
        device = self.art_embedder.device

        inputs = processor(
            text=[text], return_tensors="pt",
            padding="max_length", max_length=self.MAX_LENGTH, truncation=True,
        ).to(device)

        with torch.no_grad():
            # SigLIP's tokenizer doesn't return an attention_mask (fixed-length
            # padding, no masking by design) - spread **inputs like ArtEmbedd does,
            # rather than indexing "attention_mask" directly.
            text_out = self.art_embedder.model.text_model(**inputs)
        embed = text_out.pooler_output
        return embed / embed.norm(p=2, dim=-1, keepdim=True)

    def _encode_query_image(self, query_image: Image.Image) -> torch.Tensor:
        """(b) Frozen, L2-normalized query-image embedding - the query anchor."""
        processor = self.art_embedder.processor
        device = self.art_embedder.device

        inputs = processor(images=query_image.convert("RGB"), return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = self.art_embedder.model.vision_model(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
            )
        embed = vision_out.pooler_output
        return embed / embed.norm(p=2, dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # STEP 0: Preprocess
    # ------------------------------------------------------------------
    def _preprocess_result_image(self, image: Image.Image):
        """Patchifies the result image (x) and returns (pixel_values, pixel_mask, spatial_shapes)."""
        processor = self.art_embedder.processor
        device = self.art_embedder.device

        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
        return inputs["pixel_values"], inputs["pixel_attention_mask"], inputs["spatial_shapes"]

    def _image_embed_from_patches(self, pixel_values, pixel_attention_mask, spatial_shapes) -> torch.Tensor:
        """Vision forward pass -> normalized image embedding (gradients flow to pixel_values)."""
        with torch.set_grad_enabled(True):
            vision_out = self.art_embedder.model.vision_model(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
        embed = vision_out.pooler_output
        return embed / embed.norm(p=2, dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # STEP 1: Accumulate gradients along the straight-line path
    # ------------------------------------------------------------------
    def _accumulate_gradients(self, query_embed, pixel_values, pixel_attention_mask, spatial_shapes):
        """Returns (diff, avg_grad) = ((x - x'), average d(cos)/dxk over the path)."""
        model = self.art_embedder.model
        baseline = torch.zeros_like(pixel_values)       # x' <- zeros_like(x)
        diff = pixel_values - baseline                   # (x - x')

        total_grad = torch.zeros_like(pixel_values)
        try:
            for k in range(1, self.STEPS + 1):
                alpha = k / self.STEPS
                xk = (baseline + alpha * diff).detach().requires_grad_(True)  # interpolated image

                img_embed = self._image_embed_from_patches(xk, pixel_attention_mask, spatial_shapes)
                z = (query_embed * img_embed).sum()      # cos(img_emb(xk), query_emb)

                z.backward()
                total_grad = total_grad + xk.grad
        finally:
            # release the parameter gradients accumulated across all STEPS
            # backward passes so the shared ArtEmbedd model is left as it was found
            model.zero_grad(set_to_none=True)

        return diff, total_grad / self.STEPS

    # ------------------------------------------------------------------
    # STEP 2-3: IG formula -> saliency map
    # ------------------------------------------------------------------
    def _compute_saliency(self, diff, avg_grad, grid_hw) -> np.ndarray:
        ig = (diff * avg_grad).abs()          # STEP 2 (element-wise) + STEP 3 abs

        # ig: [1, N, patch_size*patch_size*channels]; each patch's flat vector
        # interleaves its (patch_size x patch_size) pixels with the 3 color
        # channels, so a per-patch max plays the role of "max over RGB
        # channels", extended to the patch's pixels too.
        patch_scores = ig[0].amax(dim=-1)     # [N]

        h, w = grid_hw
        saliency = patch_scores[: h * w].reshape(h, w)    # discard padded patches
        saliency = saliency.detach().cpu().numpy().astype(np.float32)

        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency

    def _overlay_heatmap(self, image: Image.Image, saliency: np.ndarray) -> Image.Image:
        """Bilinear-upsample the patch-grid saliency map back to the original image size and blend it in."""
        rgb_image = image.convert("RGB")
        w, h = rgb_image.size
        saliency_resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        base = np.array(rgb_image, dtype=np.uint8)
        blended = cv2.addWeighted(heatmap_rgb, self.HEATMAP_ALPHA, base, 1 - self.HEATMAP_ALPHA, 0)
        return Image.fromarray(blended)

    def explain(self, image: Image.Image, query) -> Image.Image:
        """Runs Integrated Gradients end-to-end and returns a saliency heatmap overlay on `image`.

        `query` is either a text string (text-query search) or a PIL Image
        (image-query search) - matching the two search modes in proto_type_main.py.
        """
        pixel_values, pixel_attention_mask, spatial_shapes = self._preprocess_result_image(image)
        grid_hw = tuple(spatial_shapes[0].tolist())

        query_embed = self._encode_text(query) if isinstance(query, str) else self._encode_query_image(query)

        diff, avg_grad = self._accumulate_gradients(query_embed, pixel_values, pixel_attention_mask, spatial_shapes)
        saliency = self._compute_saliency(diff, avg_grad, grid_hw)
        return self._overlay_heatmap(image, saliency)

    # ------------------------------------------------------------------
    # Streamlit dialog
    # ------------------------------------------------------------------
    def show_dialog(self, title: str, image: Image.Image, query):
        """Compute the Integrated Gradients saliency map and open a Streamlit dialog to display it.
        `query` is either a text string or a PIL Image (see explain())."""

        query_desc = f"the query “{query}”" if isinstance(query, str) else "the query image"

        @st.dialog(f"Integrated Gradients — {title}", width="large")
        def _render_dialog():
            with st.spinner(f"Computing Integrated Gradients ({self.STEPS} steps)…"):
                try:
                    overlay_img = self.explain(image, query)
                except Exception as e:
                    st.error(f"Could not compute Integrated Gradients: {e}")
                    return

            st.image(overlay_img, width="stretch")
            st.caption(
                f"Highlights the pixels whose values most increased the match score for {query_desc}."
            )

        _render_dialog()
