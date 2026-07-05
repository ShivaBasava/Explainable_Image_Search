"""
ALGORITHM: Grad-CAM for SigLIP2-NaFlex
    (a) text query  -> image explanation
    (b) image query -> image explanation

INPUT:
    image                # original (result) image, kept for overlay
    query                # text string, OR a query PIL image
    model                # siglip2-base-patch16-naflex (reused from ArtEmbedd)
    L                    # target layer: last encoder layer output
                         # (patch tokens, shape [N, D]) - all N are patches, no CLS

------------------------------------------------------------------
STEP 0: Preprocess
    pixel_values, pixel_mask, (h, w) <- naflex_processor(image)
        # pixel_mask marks real vs padded patches
        # (h, w) = true patch grid, varies per image - store it!
        # also store resize geometry for the final overlay

STEP 1: Forward pass (with gradients enabled)
    A <- activations of layer L                    # [N, D], hook & retain
    img_emb <- MAP_head(A) -> projection -> normalize
    query_emb <- text_encoder(text) -> normalize        # (a) EOS pooling
              <- MAP_head(query image) -> normalize     # (b) frozen anchor, no grad

    cos  <- dot(img_emb, query_emb)
    z    <- logit_scale . cos + logit_bias          # TARGET = logit for (a)
                                                    # TARGET = cos      for (b)
                                                    # (NOT sigmoid(z) - gradient dies
                                                    #  at confident matches;
                                                    #  logit_scale/bias are a positive affine
                                                    #  transform of cos, so using cos directly
                                                    #  for (b) gives the identical CAM pattern)

STEP 2: Backward pass
    G <- dz / dA                                    # [N, D], gradient at layer L

STEP 3: Channel weights  (Grad-CAM's core step)
    G <- G  where pixel_mask = 1, else 0            # drop padded patches
    alpha <- mean over real patches of G            # [D]  global-average-pooled grads

STEP 4: Weighted combination + ReLU
    for each patch i:
        cam[i] <- ReLU( sum_d  alpha[d] . A[i, d] ) # [N]

STEP 5: Reshape & upsample
    cam <- cam[0 : h*w]                             # discard padded positions
    cam <- reshape(cam, [h, w])                     # naflex grid, NOT fixed 16x16
    cam <- normalize to [0, 1]                      # (cam - min) / (max - min)
    cam <- bilinear_upsample(cam -> resized image size)
    cam <- invert resize geometry -> original image size

OUTPUT:
    heatmap overlay on original image
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class GradCAMExplainer:
    """Grad-CAM explanation for SigLIP2-NaFlex, for both search modes:
    text query -> image explanation, and image query -> image explanation.

    Highlights which image patches most increased the similarity between the
    query (text or image) and a result image, using the gradient of the
    match score with respect to the patch-token activations of the last
    vision-encoder layer.
    """

    MAX_LENGTH = 64             # matches ArtEmbedd's text tokenization length
    HEATMAP_ALPHA = 0.45        # overlay blend weight

    def __init__(self, art_embedder):
        """art_embedder: an ArtEmbedd instance. Reused so Grad-CAM runs on the
        exact same loaded SigLIP2 weights/processor/device as the search index,
        instead of loading a second copy of the model."""
        self.art_embedder = art_embedder

    def _get_vision_target_layer(self):
        """Last vision-encoder layer (L): output is patch-token activations, no CLS."""
        return self.art_embedder.model.vision_model.encoder.layers[-1]

    def _hook_target_layer(self):
        """Registers the activation-retaining forward hook on layer L.
        Returns (handle, activations_dict); read activations_dict["A"] after the
        forward pass, then always call handle.remove()."""
        activations = {}

        def _retain_activations(module, layer_input, layer_output):
            layer_output.retain_grad()
            activations["A"] = layer_output

        handle = self._get_vision_target_layer().register_forward_hook(_retain_activations)
        return handle, activations

    def _forward_logit_text(self, image: Image.Image, text: str):
        """
        (a) text query -> image explanation.
        STEP 0-1: preprocess + forward pass with gradients enabled.

        Returns (z, activations, grid_hw, pixel_mask):
            z           - SigLIP2 logit (scalar tensor) for (image, text)
            activations - patch-token activations A of the target layer [1, N, D]
            grid_hw     - true (patch_h, patch_w) grid for this image (NaFlex, varies per image)
            pixel_mask  - pixel_attention_mask [1, N] (1 = real patch, 0 = padding)
        """
        processor = self.art_embedder.processor
        device = self.art_embedder.device

        inputs = processor(
            images=image.convert("RGB"), text=[text], return_tensors="pt",
            padding="max_length", max_length=self.MAX_LENGTH, truncation=True,
        ).to(device)

        handle, activations = self._hook_target_layer()
        try:
            with torch.set_grad_enabled(True):
                outputs = self.art_embedder.model(**inputs)
        finally:
            handle.remove()

        z = outputs.logits_per_text[0, 0]   # TARGET = logit (NOT sigmoid(z))
        grid_hw = tuple(inputs["spatial_shapes"][0].tolist())
        pixel_mask = inputs["pixel_attention_mask"]

        return z, activations["A"], grid_hw, pixel_mask

    def _encode_query_image(self, query_image: Image.Image) -> torch.Tensor:
        """Embeds the query image as a frozen anchor (no gradient needed) -
        gradients only need to flow through the *result* image being explained."""
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

    def _forward_logit_image(self, query_image: Image.Image, result_image: Image.Image):
        """
        (b) image query -> image explanation.
        STEP 0-1: preprocess + forward pass with gradients enabled, on the
        result image only; the query image is a frozen (no-grad) anchor.

        Returns (z, activations, grid_hw, pixel_mask) - same shape as
        _forward_logit_text, so both paths share _compute_cam/_overlay_heatmap.
        """
        processor = self.art_embedder.processor
        device = self.art_embedder.device

        query_embed = self._encode_query_image(query_image)

        inputs = processor(images=result_image.convert("RGB"), return_tensors="pt").to(device)

        handle, activations = self._hook_target_layer()
        try:
            with torch.set_grad_enabled(True):
                vision_out = self.art_embedder.model.vision_model(
                    pixel_values=inputs["pixel_values"],
                    pixel_attention_mask=inputs["pixel_attention_mask"],
                    spatial_shapes=inputs["spatial_shapes"],
                )
        finally:
            handle.remove()

        result_embed = vision_out.pooler_output
        result_embed = result_embed / result_embed.norm(p=2, dim=-1, keepdim=True)

        z = (query_embed * result_embed).sum()   # TARGET = cosine similarity
        grid_hw = tuple(inputs["spatial_shapes"][0].tolist())
        pixel_mask = inputs["pixel_attention_mask"]

        return z, activations["A"], grid_hw, pixel_mask

    def _compute_cam(self, z, activations, grid_hw, pixel_mask) -> np.ndarray:
        """STEP 2-5: backward pass -> channel weights -> weighted ReLU -> normalized [h, w] map."""
        model = self.art_embedder.model
        model.zero_grad(set_to_none=True)
        try:
            z.backward()

            grad = activations.grad                 # G = dz/dA   [1, N, D]
            real_mask = pixel_mask[0].bool()         # [N]

            real_grad = grad[0][real_mask]           # drop padded patches
            alpha = real_grad.mean(dim=0)            # [D] global-average-pooled grads

            real_activations = activations[0][real_mask]              # [n_real, D]
            cam = F.relu((alpha * real_activations).sum(dim=-1))      # [n_real]

            h, w = grid_hw
            cam = cam[: h * w].reshape(h, w)
            cam = cam.detach().cpu().numpy().astype(np.float32)
        finally:
            # release the parameter gradients accumulated by backward() so the
            # shared ArtEmbedd model is left as it was found (no lingering .grad)
            model.zero_grad(set_to_none=True)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def _overlay_heatmap(self, image: Image.Image, cam: np.ndarray) -> Image.Image:
        """Bilinear-upsample the patch-grid CAM back to the original image size and blend it in."""
        rgb_image = image.convert("RGB")
        w, h = rgb_image.size
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        base = np.array(rgb_image, dtype=np.uint8)
        blended = cv2.addWeighted(heatmap_rgb, self.HEATMAP_ALPHA, base, 1 - self.HEATMAP_ALPHA, 0)
        return Image.fromarray(blended)

    def explain(self, image: Image.Image, query) -> Image.Image:
        """Runs Grad-CAM end-to-end and returns a heatmap overlay on `image` (the result).

        `query` is either a text string (text-query search) or a PIL Image
        (image-query search) - matching the two search modes in proto_type_main.py.
        """
        if isinstance(query, str):
            z, activations, grid_hw, pixel_mask = self._forward_logit_text(image, query)
        else:
            z, activations, grid_hw, pixel_mask = self._forward_logit_image(query, image)

        cam = self._compute_cam(z, activations, grid_hw, pixel_mask)
        return self._overlay_heatmap(image, cam)

    # ------------------------------------------------------------------
    # Streamlit dialog
    # ------------------------------------------------------------------
    def show_dialog(self, title: str, image: Image.Image, query):
        """Compute the Grad-CAM heatmap and open a Streamlit dialog to display it.
        `query` is either a text string or a PIL Image (see explain())."""

        query_desc = f"the query “{query}”" if isinstance(query, str) else "the query image"

        @st.dialog(f"Grad-CAM — {title}", width="large")
        def _render_dialog():
            with st.spinner("Computing Grad-CAM…"):
                try:
                    overlay_img = self.explain(image, query)
                except Exception as e:
                    st.error(f"Could not compute Grad-CAM: {e}")
                    return

            st.image(overlay_img, width="stretch")
            st.caption(f"Highlights the image regions that most increased the match score for {query_desc}.")

        _render_dialog()
