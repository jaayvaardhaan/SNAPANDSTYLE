# Datasets

- Root CSV files are the active cleaned, filtered and tagged catalogs.
- `kaggle_fashion/styles.csv` is the original catalog metadata.
- `kaggle_fashion/images/` contains the active product JPEGs served by the backend.
- `processed/` contains derived DeepFashion data and sample crops.
- `archive/metadata/` preserves duplicate generated CSVs from the download.
- `archive/myntradataset/` preserves the duplicate original dataset download.
- `archive/__MACOSX/` preserves Mac archive metadata.

Recommendation preparation scripts live in `recommendation_engine/scripts/`. They resolve these paths from the project root. DeepFashion scripts require an additional `datasets/deepfashion2/` source dataset; it is separate from the active Kaggle catalog.
