import json
import os
import pathlib
from typing import Generator

from datasets import load_dataset


def dataset(path):
    assert len(os.listdir(path)) > 0
    if "model" not in os.listdir(path):
        os.mkdir(f"{path}/model")


def collect_captioned_images(root_folder: str) -> Generator[tuple, None, None]:
    for directory, _, filenames in os.walk(root_folder):
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_filenames = [f for f in filenames if os.path.splitext(f)[1] in image_extensions]
        for image_filename in image_filenames:
            caption_filename = os.path.splitext(image_filename)[0] + '.txt'
            caption_path = os.path.join(directory, caption_filename)
            if not os.path.exists(caption_path):
                continue

            with open(caption_path, 'r') as f:
                caption = f.read().replace('\n', ' ')

                image_path = os.path.join(directory, image_filename)
                yield image_path, caption


def convert_text_image_pairs_to_huggingface_json(root_folder, out_json):
    out_folder = os.path.dirname(root_folder)
    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        written_count = 0
        for image_path, caption in collect_captioned_images(root_folder):
            line_dict = {"image": image_path, "caption": caption}
            json_line = json.dumps(line_dict, indent=None, separators=(",", ":"))
            f.write(json_line + "\n")
            written_count += 1
        print(f"wrote {written_count} lines to {out_json}")


if __name__ == "__main__":
    DATASET_PATH = os.environ["DATASET_PATH"] if "DATASET_PATH" in os.environ else "./finetuned/dataset"
    OUT_JSON = os.environ["OUT_JSON"] if "OUT_JSON" in os.environ else "./finetuned/src.json"
    REPO_ID = os.environ["REPO_ID"] if "REPO_ID" in os.environ else "openai/clip-vit-large-patch14-336"
    OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"] if "OUTPUT_FOLDER" in os.environ else "./finetuned/model"
    BATCH_SIZE = int(os.environ["BATCH_SIZE"]) if "BATCH_SIZE" in os.environ else 32
    NUM_TRAIN_EPOCHS = int(os.environ["NUM_TRAIN_EPOCHS"]) if "NUM_TRAIN_EPOCHS" in os.environ else 3

    dataset(path=DATASET_PATH)  # detecting dataset

    convert_text_image_pairs_to_huggingface_json(DATASET_PATH, OUT_JSON)  # creating dataset

    dataset = load_dataset("json", data_files=OUT_JSON)  # test loading it back in
