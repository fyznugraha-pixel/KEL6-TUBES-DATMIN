import json
import base64
from pathlib import Path

NOTEBOOK_PATH = "KELOMPOK_6.ipynb"
OUTPUT_DIR = Path("static/img")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

selected_images = [
    {
        "filename": "01_distribusi_target.png",
        "cell_index": 6,
        "output_index": 2
    },
    {
        "filename": "02_heatmap_korelasi.png",
        "cell_index": 14,
        "output_index": 0
    },
    {
        "filename": "03_feature_importance.png",
        "cell_index": 20,
        "output_index": 1
    },
    {
        "filename": "04_elbow_method.png",
        "cell_index": 32,
        "output_index": 0
    },
    {
        "filename": "05_silhouette_score.png",
        "cell_index": 32,
        "output_index": 1
    },
    {
        "filename": "06_cluster_pca.png",
        "cell_index": 34,
        "output_index": 0
    },
    {
        "filename": "07_confusion_matrix_random_forest.png",
        "cell_index": 38,
        "output_index": 5
    }
]

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as file:
    notebook = json.load(file)

for item in selected_images:
    cell = notebook["cells"][item["cell_index"]]
    output = cell["outputs"][item["output_index"]]

    if "data" not in output or "image/png" not in output["data"]:
        print(f"Gagal export: {item['filename']} tidak punya image/png")
        continue

    image_data = output["data"]["image/png"]

    if isinstance(image_data, list):
        image_data = "".join(image_data)

    image_bytes = base64.b64decode(image_data)

    output_path = OUTPUT_DIR / item["filename"]

    with open(output_path, "wb") as img_file:
        img_file.write(image_bytes)

    print(f"Berhasil export: {output_path}")

print("\nSelesai. Semua visualisasi terpilih sudah masuk ke folder static/img.")