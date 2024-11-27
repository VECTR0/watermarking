from kagglehub import dataset_download

# Datasets downloads here:
# ~/.cache/kagglehub/datasets/kamino/largescale-common-watermark-dataset/versions/1
# ~/.cache/kagglehub/datasets/felicepollano/watermarked-not-watermarked-images

KAGLE_DOWNLOAD_PATHS = [
    {
        "name": "Large-scale Common Watermark Dataset, 2.32GB",
        "link": "https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset",
        "download": "kamino/largescale-common-watermark-dataset",
    },
    {
        "name": "Watermarked / Not watermarked images 754.3 MB",
        "link": "https://www.kaggle.com/datasets/felicepollano/watermarked-not-watermarked-images",
        "download": "felicepollano/watermarked-not-watermarked-images",
    },
]


for k in KAGLE_DOWNLOAD_PATHS:
    name, link, download = k["name"], k["link"], k["download"]
    path = dataset_download(download)
    print(f"Dataset:{name},\nLink:{link},\nPath:{path}")
