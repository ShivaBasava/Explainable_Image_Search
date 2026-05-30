"""

for handling local index
"""
import os
import json
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from config import get_config
INDEX_FILE = get_config("INDEX_FILE")
META_FILE = get_config("META_FILE")
FILE_EXT = tuple(get_config("FILE_EXT"))



class ArtIndexer:
    """class for building, loading & appending Index
     perform_similarity_search() - Webapp-1 search function"""
 
    def __init__(self, index_file=INDEX_FILE, meta_file=META_FILE):
        self.index_file = index_file
        self.meta_file = meta_file

    def _new_index(self, dim: int):
        return faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def load_state(self):
        if os.path.exists(self.index_file):
            index = faiss.read_index(self.index_file)
        else:
            index = None

        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = { "items": [],   "next_id": 0,   "url_to_id": {}   }

        return index, meta

    def save_state(self, index, meta):
        faiss.write_index(index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def create_or_append_index(self, embeddings, records):
        """
        embedding & storing index  """
        
        if not embeddings:
            raise ValueError("No embeddings provided to indexer.")

        index, meta = self.load_state()

        new_vecs = []
        new_ids = []
        new_items = []

        for emb, rec in zip(embeddings, records):
            url = rec.get("image_url", "")
            if not url:
                continue
            if url in meta["url_to_id"]:
                continue

            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if vec.size == 0 or not np.isfinite(vec).all():
                continue

            if index is not None and vec.shape[0] != index.d:
                raise ValueError(
                    f"Embedding dimension mismatch for {url}: got {vec.shape[0]}, expected {index.d}"
                )

            if index is None:
                index = self._new_index(vec.shape[0])

            item_id = int(meta["next_id"])
            meta["next_id"] += 1

            new_vecs.append(vec)
            new_ids.append(item_id)
            new_item = dict(rec)
            new_item["id"] = item_id
            new_items.append(new_item)
            meta["url_to_id"][url] = item_id

        if not new_vecs:
            self.save_state(index, meta)
            return index, meta

        vectors = np.vstack(new_vecs).astype(np.float32)
        ids = np.asarray(new_ids, dtype=np.int64)
        faiss.normalize_L2(vectors)
        
        index.add_with_ids(vectors, ids)
        meta["items"].extend(new_items)
        self.save_state(index, meta)

        return index, meta


    def load_faiss_index(self):
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Missing index file: {self.index_file}")
        index = faiss.read_index(self.index_file)

        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {"items": [], "next_id": 0, "url_to_id": {}}

        return index, meta


    def perform_similarity_search(self, query, embed_fn, index, meta, top_k=5):
        
        if isinstance(query, str) and query.lower().endswith( FILE_EXT ):
            query = Image.open(query).convert("RGB")

        query_vec = embed_fn(query)
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)

        faiss.normalize_L2(query_vec)

        scores, ids = index.search(query_vec, top_k)

        id_to_item = {int(item["id"]): item for item in meta.get("items", [])}

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            item = id_to_item.get(int(idx))
            
            if not item:
                continue
                
            cosine_score = float(score)
            similrity = round((cosine_score + 1.0) / 2.0 * 100.0, 1)
            
            results.append({
                "id": int(idx),    "qid": item.get("qid", ""),
                "image_url": item.get("image_url", ""),   "title": item.get("title", ""),    
                "artist": item.get("artist", ""),     "depicts": item.get("depicts", ""),
                "description": item.get("description", ""),
                "wikidata_url": item.get("wikidata_url", ""),
                "cosine_score": cosine_score,   "similarity": similrity
            })

        return query, results
