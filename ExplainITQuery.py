"""
Part of WebApp-1

Explainability implemented:
- Similarity score  (cosine similarity, displayed as %)
- Explanability method for both text & image query
0. FOr Text query & Image query: a global score as a Similarity
1. Image query: patch occlusion sensitivity - Region Importance
- which regions of the image were most important for the match.
2. Text query: leave-one-out token importance sensitivity : Extracted keyword Importance
- which words in the query were most important for the match.
"""

import io
import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None


import string
import streamlit as st
from config import get_config

class ExplainITQuery:

    def __init__(self):

        self.WIKI_HEADERS = get_config("WIKI_HEADERS")
        self.TIMEOUT = get_config("TIMEOUT")
        self.CUSTOM_STOP_WORDS = set(get_config("CUSTOM_STOP_WORDS", []))
        
        #for image patch importance
        self.OCCLUSION_GREY: int = 114
 
        # image partially visible through the overlay.
        self.HEATMAP_ALPHA_MIN: int = 30
        self.HEATMAP_ALPHA_MAX: int = 220

        self.COLOUR_POSITIVE = (220, 50,  50)   # red
        self.COLOUR_NEGATIVE = (50,  80, 220)   # blue


    def _cosine(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Dot product, two L2-normalised 
        vectors, leads to cosine similarity."""
        return float(np.dot(vec_a.ravel(), vec_b.ravel()))


    def explain_image_query( self, query_image: Image.Image,
            result_image_url: str, call_embed_fn, grid: int,
        ) -> tuple[Image.Image, list[float]]:
        """
        for an image query -Region Importance map
            for result image.
        """
        try:
            resp = requests.get(result_image_url, timeout=self.TIMEOUT, headers=self.WIKI_HEADERS,
                                stream=True, allow_redirects=True)
            resp.raise_for_status()
            result_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            return query_image, []

        query_rgb = query_image.convert("RGB")

        # baseline similarity using untouched vectors
        vec_query_base = call_embed_fn(query_rgb)
        vec_result_base = call_embed_fn(result_img)
        baseline = self._cosine(vec_query_base, vec_result_base)

        
        # active_img = query_rgb
        active_img = result_img
        w, h = active_img.size
        pw, ph = w // grid, h // grid
        active_arr = np.array(active_img, dtype=np.uint8)

        grey_fill = np.full(3, self.OCCLUSION_GREY, dtype=np.uint8)
        patch_importances = []

        # occlusion loop
        for row_i in range(grid):
            for col_i in range(grid):
                ablated = active_arr.copy()
                y0, y1 = row_i * ph, (row_i + 1) * ph
                x0, x1 = col_i * pw, (col_i + 1) * pw
                ablated[y0:y1, x0:x1, :] = grey_fill

                ablated_pil = Image.fromarray(ablated, mode="RGB")

                # re-calculate similarity based on each image is modified
                current_sim = self._cosine(vec_query_base, call_embed_fn(ablated_pil))

                drop = baseline - current_sim
                patch_importances.append(float(drop))

        # color normalization
        arr = np.array(patch_importances, dtype=np.float32)
        pos_max = arr.max() if arr.max() > 0 else 1.0
        neg_min = arr.min() if arr.min() < 0 else -1.0

        base_rgba = active_img.convert("RGBA")
        tint_layer = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(tint_layer)

        for i, importance in enumerate(patch_importances):
            row_i, col_i = divmod(i, grid)
            y0, y1 = row_i * ph, (row_i + 1) * ph
            x0, x1 = col_i * pw, (col_i + 1) * pw

            if importance >= 0:
                norm_val = importance / pos_max
                colour = self.COLOUR_POSITIVE
            else:
                norm_val = abs(importance) / abs(neg_min)
                colour = self.COLOUR_NEGATIVE

            alpha = int(self.HEATMAP_ALPHA_MIN + norm_val * (self.HEATMAP_ALPHA_MAX - self.HEATMAP_ALPHA_MIN))
            draw.rectangle([x0, y0, x1, y1], fill=(*colour, alpha))


        max_idx = patch_importances.index(max(patch_importances))
        min_idx = patch_importances.index(min(patch_importances))
        
        max_row, max_col = (max_idx // grid) + 1, (max_idx % grid) + 1
        min_row, min_col = (min_idx // grid) + 1, (min_idx % grid) + 1

        patch_metrics = {
            "highest": {
                "importance": patch_importances[max_idx],
                "row": max_row,
                "col": max_col
            },
            "lowest": {
                "importance": patch_importances[min_idx],
                "row": min_row,
                "col": min_col
            }
        }

        return Image.alpha_composite(base_rgba, tint_layer).convert("RGB"), patch_metrics


    def explain_text_query(self, query_text: str, result_image_url: str, embed_fn, top_n: int) -> list[dict]:
    
        tokens = query_text.split()
        if not tokens:
            return []

        stop_words = set(self.CUSTOM_STOP_WORDS)

        try:
            resp = requests.get(result_image_url, timeout=self.TIMEOUT, headers=self.WIKI_HEADERS, stream=True)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            result_vec = embed_fn(img)
        except Exception:
            return []

        full_vec = embed_fn(query_text) 
        baseline_score = self._cosine(full_vec, result_vec)

        importances = []
        
        for i, tok in enumerate(tokens):
            # punctuation and lowercase
            
            cleaned_tok = tok.lower().strip(string.punctuation)
            
            
            if not cleaned_tok:
                continue
                

            if cleaned_tok in stop_words:
                continue  
                
            # to build ablated sentence
            ablated_list = [t for j, t in enumerate(tokens) if j != i]
            ablated = " ".join(ablated_list) if ablated_list else "."
            
            ablated_score = self._cosine(embed_fn(ablated), result_vec)
            score_drop = baseline_score - ablated_score
            
            importances.append({
                "token": cleaned_tok,
                "token_idx": f"{i}_{cleaned_tok}",
                "importance": round(score_drop, 4)
            })
    
        importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
        
        return importances[:top_n]


