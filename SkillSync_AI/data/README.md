# Dataset information

## Bundled fallback dataset
`dataset.csv` is a small, hand-curated educational fallback catalog of career-role/skill examples. It is included so the project can train and run offline immediately. It is **not** represented as a collected real-world job-postings dataset, and model metrics from it must be described as prototype/demonstration metrics.

## Public dataset option
A genuine public career-path dataset is listed on Mendeley Data:

- Title: careerpath / Career Path Prediction
- Landing page: https://data.mendeley.com/datasets/cns7h4b9g7/1
- Dataset description: 15,000 entries; profile attributes and a suggested career path; 19 features.
- License shown on the listing: CC BY 4.0.
- Citation: vaishu, vaishu (2025), “careerpath”, Mendeley Data, V1, DOI: 10.17632/cns7h4b9g7.1.

Run `python -m src.dataset_download` to attempt download to `career_path_public.csv`. The downloader uses the published Mendeley public-file endpoint. The public file's schema can change; the loader accepts common `career`/`role`/`suggested career path` and `skills` column names. If download or schema validation fails, delete/rename the downloaded file and the bundled fallback is used.

The app automatically prefers `career_path_public.csv` when present and valid; otherwise it uses `dataset.csv`.
