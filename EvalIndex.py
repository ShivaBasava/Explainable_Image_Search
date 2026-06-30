import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import requests
import io
from tqdm import tqdm


from ArtEmbedd import ArtEmbedd
from ArtIndexer import ArtIndexer

#loading project configuration
CONFIG_PATH = '~/Explainable_Image_Search/app_config.toml'

try:
    import tomllib
    with open(CONFIG_PATH, "rb") as f:
        CONFIG = tomllib.load(f)
except ImportError:
    try:
        import toml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = toml.load(f)
    except ImportError:
        print("Error: Please run 'pip install toml' to parse configurations.")
        sys.exit(1)
except FileNotFoundError:
    print("Error: 'app_config.toml' file is missing.")
    sys.exit(1)



class EvalIndex:
    """evaluate the local .index 
    against a ground truth excel file"""

    def __init__(self):
        self.embedder = ArtEmbedd()
        self.indexer = ArtIndexer()
        self.index, self.meta = self.indexer.load_faiss_index()
        self.indexed_qids = {item["qid"] for item in self.meta["items"]}
        self.qid_to_item = {item["qid"]: item for item in self.meta["items"]}


    def retrieve_by_qid(self, query_qid, max_needed=25):
        """Retrieve candidates- handles self-exclusion of QID."""
        if query_qid not in self.qid_to_item:
            return []

        headers = dict(CONFIG.get("WIKI_HEADERS", {}))
        timeout = int(CONFIG.get("TIMEOUT", 20))
        image_url = self.qid_to_item[query_qid]["image_url"]

        max_retries = 3
        pause = 2.0
        image = None

        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, headers=headers, timeout=timeout)
                if response.status_code == 429:
                    time.sleep(pause * (attempt + 1))
                    continue

                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n[WARN] Failed to download image for QID {query_qid}: {e}")
                    return []
                
                time.sleep(pause * (attempt + 1))

        if image is None:
            return []

        # elements to account for self-exclusion
        _, results = self.indexer.perform_similarity_search(
                image, self.embedder.get_embedding, self.index, self.meta,
                top_k=max_needed )

        # Excluding self-matches
        return [r["qid"] for r in results if r["qid"] != query_qid]

    def compute_metrics(self, retrieved, relevant, k):
        # slicing K elements for calculation
        retrieved_k = retrieved[:k]
        hits = sum(1 for qid in retrieved_k if qid in relevant)

        precision = hits / k if k > 0 else 0.0
        recall = hits / len(relevant) if len(relevant) > 0 else 0.0

        running = 0
        ap_sum = 0.0
        for rank, qid in enumerate(retrieved_k, 1):
            if qid in relevant:
                running += 1
                ap_sum += running / rank
        ap = ap_sum / len(relevant) if relevant else 0.0

        return precision, recall, ap


    def evaluate_all_k_optimized(self, excel_path, k_list=[5, 10, 15, 20], sample_columns=None, random_seed=111):
        """Evaluates all requested K, 
        in a single pass.        """

        df = pd.read_excel(excel_path, header=None).fillna("")
        total_cols = df.shape[1]
        print(f"Ground truth has {total_cols} query columns and {df.shape[0]-1} rows.")

        if sample_columns is not None and sample_columns < total_cols:
            np.random.seed(random_seed)
            eval_cols = np.random.choice(total_cols, size=sample_columns, replace=False)
            eval_cols = sorted(eval_cols)
        
            print(f"Randomly sampled {len(eval_cols)} columns for evaluation.")
        else:
            eval_cols = range(total_cols)

        # preparing final dictionary
        results_by_k = {k: {"P": [], "R": [], "AP": []} for k in k_list}
        skipped_no_header = 0
        skipped_no_targets = 0
        max_k = max(k_list)

        for col in tqdm(eval_cols, desc="Evaluating columns"):
            query_qid = str(df.iloc[0, col]).strip()
            if not query_qid or query_qid not in self.indexed_qids:
                skipped_no_header += 1
                continue

            raw_rows = [str(df.iloc[row, col]).strip() for row in range(1, df.shape[0])]
            relevant_qids = {qid for qid in raw_rows if qid and qid in self.indexed_qids}

            if not relevant_qids:
                skipped_no_targets += 1
                continue

            retrieved_qids = self.retrieve_by_qid(query_qid, max_needed=max_k)
            if not retrieved_qids:
                continue

            for k in k_list:
                p, r, ap = self.compute_metrics(retrieved_qids, relevant_qids, k)
                
                results_by_k[k]["P"].append(p)
                results_by_k[k]["R"].append(r)
                results_by_k[k]["AP"].append(ap)

        final_summary = {}
        for k in k_list:
            final_summary[k] = {
                "Ground truth file": excel_path, "Top-K": k, "Evaluated columns": len(results_by_k[k]["P"]),
                f"Precision@{k}": np.mean(results_by_k[k]["P"]) if results_by_k[k]["P"] else 0.0,
                f"Recall@{k}": np.mean(results_by_k[k]["R"]) if results_by_k[k]["R"] else 0.0,
                f"mAP@{k}": np.mean(results_by_k[k]["AP"]) if results_by_k[k]["AP"] else 0.0,

                "Skipped (no header in index)": skipped_no_header,
                "Skipped (no relevant targets)": skipped_no_targets,
            }
        return final_summary

if __name__ == "__main__":
    #given ground truth file, path to test.xlsx
    GROUND_TRUTH_PATH = "~/test.xlsx"
    evaluator = EvalIndex()

    # all Top-K in parallel, evaluating 100 randomsample col 
    #from the test.xlsx on the local index
    all_results = evaluator.evaluate_all_k_optimized(
        GROUND_TRUTH_PATH, k_list=[5, 10, 15, 20], sample_columns=100, 
        random_seed=111 #for reproducibility    
        )

    for k in all_results:
        print(f"-------\nEvaluation results (Top-{k}):")
        for metric, value in all_results[k].items():
            print(f"{metric}: {value}")

