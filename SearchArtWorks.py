import requests, re
import pandas as pd
from config import get_config

WIKI_HEADERS = get_config("WIKI_HEADERS")

SPARQL_ENDPOINT = get_config("SPARQL_ENDPOINT")
TIMEOUT = get_config("TIMEOUT")


class SearchArtWorks:
    """Searches + processes artwork data from Wikidata via SPARQL.
    """

    def __init__(self):
        self.SPARQL_ENDPOINT = SPARQL_ENDPOINT
        self.HEADERS = WIKI_HEADERS
        self.RESPONSE_STATUS_CODE = 200

    #  SparQL query                                                       
    def search_wikidata_artworks(self, search_term: str, limit: int, is_qid_search: bool = False):
        """either free-text searches INNER_LIMIT
         or direct QID list lookups
        """
        if is_qid_search:
            # inputs like "Q123 Q456" into "wd:Q123 wd:Q456"
            qids = search_term.replace(",", " ").split()
            formatted_qids = " ".join([f"wd:{q.strip()}" for q in qids if q.strip()])
            
            # QID lookup block
            target_search_block = f"VALUES ?artwork {{ {formatted_qids} }}"
            
        else:
            
            if search_term.startswith('"') and search_term.endswith('"'):
                gsr_search = search_term          # exact phrase
            else:
            
                words = search_term.split()
                gsr_search = " ".join(words) if len(words) > 1 else search_term

            INNER_LIMIT = 30
            
            target_search_block = f"""
             SERVICE wikibase:mwapi {{
               bd:serviceParam wikibase:endpoint "www.wikidata.org";
                               wikibase:api "Generator";
                               mwapi:generator "search";
                               mwapi:gsrsearch "{gsr_search}";
                               mwapi:gsrlimit "{INNER_LIMIT}".
              ?artwork wikibase:apiOutputItem mwapi:title.
            }}
            """
            
        QUERY = f"""
           SELECT DISTINCT ?artwork ?artworkLabel ?image ?description ?artist ?artistLabel
           (GROUP_CONCAT(DISTINCT ?enLabel; SEPARATOR=", ") AS ?depictsEN)
           WHERE {{
             
             {target_search_block}
             
             ?artwork wdt:P31/wdt:P279* wd:Q3305213;
                      wdt:P18 ?image.
             OPTIONAL {{
               ?artwork schema:description ?description.
               FILTER(LANG(?description) = "en")
             }}
             OPTIONAL {{ 
               ?artwork wdt:P180 ?depicts. 
               OPTIONAL {{ ?depicts rdfs:label ?enLabel. FILTER(LANG(?enLabel) = "en") }}
             }}
             OPTIONAL {{ ?artwork wdt:P170 ?artist. }}
             SERVICE wikibase:label {{
               bd:serviceParam wikibase:language "en".
               ?artwork rdfs:label ?artworkLabel.
               ?artist rdfs:label ?artistLabel.
             }}
           }}
           GROUP BY ?artwork ?artworkLabel ?image ?description ?artist ?artistLabel
           LIMIT {limit}
        """

        try:
            response = requests.post(
                self.SPARQL_ENDPOINT,
                data=QUERY,
                headers=self.HEADERS,
                timeout=TIMEOUT,
            )
            if response.status_code == self.RESPONSE_STATUS_CODE:
                return response.json()
            else:
                raise RuntimeError(
                    f"SPARQL Error {response.status_code}: {response.text[:200]}"
                )
        except requests.exceptions.Timeout:

            raise RuntimeError("Query timed out — try a simpler search term.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Connection error: {e}") from e

    def _strip_html(self, word):

        return re.sub(r'<[^>]+>', '', word).lower()


    def process_sparql_results(self, sparql_results) -> pd.DataFrame:
        """Convert SPARQL JSON response to a DataFrame."""
        if not sparql_results or "results" not in sparql_results:
            return pd.DataFrame()

        processed = []
        seen_qids = set()

        for item in sparql_results["results"]["bindings"]:

            artwork_uri = item.get("artwork", {}).get("value", "")
            q_id = artwork_uri.split("/")[-1] if "/" in artwork_uri else artwork_uri
            if q_id in seen_qids:
                continue
            seen_qids.add(q_id)

            raw_artist = item.get("artistLabel", {}).get("value", "Unknown")
            artist = "Unknown" if raw_artist.startswith("http") else raw_artist
            processed.append({
                "id":           q_id,
                "title":        self._strip_html(item.get("artworkLabel",  {}).get("value", "Unknown")),
                "image_url":    item.get("image",        {}).get("value", ""),
                "description":  item.get("description",  {}).get("value", ""),
                "artist":       self._strip_html(artist),
                "depicts":       item.get("depictsEN", {}).get("value", ""),
                "wikidata_url": artwork_uri,
            })

        return pd.DataFrame(processed)


