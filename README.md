### README for Explainable-Image-Search/
# This folder has the following files potentially for 2 mini WebApp-

## ( 1 ) Explainable-Image-Search/build_index.py

- "Streamlit UI for Build Index" [ Phase-0 | WebApp-0 ]

pip install streamlit

run script as -

streamlit run build_index.py --logger.level=debug


## ( 2 ) Explainable-Image-Search/proto_type_main.py

- "Streamlit UI for Explainable Multimodal Search" [ Phase-1 | WebApp-1 ]

pip install streamlit

run script as -

streamlit run proto_type_main.py --logger.level=debug


## ( 3 ) Explainable-Image-Search/SearchArtWorks.py

- Class ```SearchArtWorks``` for handling Wikidata SPARQL queries and artwork data processing
- This implementation which is data relevant ( quite similar to art-platform platform ) is a re-usable service instead of a stand-alone.


## ( 4 ) Explainable-Image-Search/ArtIndexer.py

- Class ```ArtIndexer``` for building, loading & appending Index
    - Also perform_similarity_search() (via FAISS) 


## ( 5 ) Explainable-Image-Search/ArtEmbedd.py

- Class ```ArtEmbedd``` for embedding wrapper via Model ( dual-encoders ).

## Explainable-Image-Search/requirements.txt

- Possible Python packages to run project


--
## Steps to run project locally
- Clone the project repository
	```git clone https://github.com/ShivaBasava/Explainable-Image-Search.git```

- ```cd Explainable-Image-Search```

- Create python environment with Python 3.11.11 version.

- Install packages from the Explainable-Image-Search/requirements.txt

- There are 2 mini-webapp build via Streamlit,
-- webApp 0: for build_index.py
-- webApp 1: proto_type_main.py for exlpainable index search (via Faiss + L2 norm)
	-- this requires files `demo_data/`

	```streamlit run proto_type_main.py --logger.level=debug```

	-- if not, one could build your their local index from webApp-0

	```streamlit run build_index.py --logger.level=debug```
	after this step, place the fresh files (.index & meta.json) to demo_data/ & updated app_config.toml path, later run webApp-1.


## NOTE - Multimodal - > Text and Image based query search.
