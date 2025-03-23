from datasets import Dataset
import pandas as pd

# dataset.json'ı oku
with open("risale-sohbet-turkish/dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Pandas DataFrame'e çevir
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Parquet olarak kaydet
dataset.to_parquet("risale-sohbet-turkish/dataset.parquet")

# Hub'a yükle
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj="risale-sohbet-turkish/dataset.parquet",
    path_in_repo="dataset.parquet",
    repo_id=f"{HF_USERNAME}/{REPO_NAME}",
    repo_type="dataset"
)