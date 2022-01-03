from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import requests
from tqdm import tqdm
import zipfile
import json
from PIL import Image
import PIL
import random
from imagehash import phash
import shutil
import threading
from dataset_utils import download_mp, unzip


class VisualGenomeDownloader:

    IMAGE_URLS = [
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
    ]
    IMAGE_OUT_DIRS = ["VG_100K", "VG_100K_2"]
    IMAGE_METADATA_URLS = [
        "https://visualgenome.org/static/data/dataset/image_data.json.zip"
    ]
    REGION_DESCRIPTIONS = [
        "https://visualgenome.org/static/data/dataset/region_descriptions.json.zip"
    ]
    QUESTION_ANSWERS = [
        "https://visualgenome.org/static/data/dataset/question_answers.json.zip"
    ]
    OBJECTS = ["https://visualgenome.org/static/data/dataset/objects.json.zip"]
    ATTRIBUTES = ["https://visualgenome.org/static/data/dataset/attributes.json.zip"]
    RELATIONSHIPS = [
        "https://visualgenome.org/static/data/dataset/relationships.json.zip"
    ]

    ALL_URLS = (
        IMAGE_METADATA_URLS
        + REGION_DESCRIPTIONS
        + QUESTION_ANSWERS
        + OBJECTS
        + ATTRIBUTES
        + RELATIONSHIPS
    )

    def __init__(self, data_dir):
        self.DATA_DIR = Path(data_dir)
        self.DATA_DIR.mkdir(exist_ok=True, parents=True)
        (
            self.attributes,
            self.objects,
            self.qa,
            self.region_descriptions,
            self.relationships,
            self.image_metadata,
            self.IMAGE_IDS,
            self.ID_TO_PATH_MAP,
            self.INDEX_TO_ID_MAP,
            self.ID_TO_INDEX_MAP,
        ) = (None,) * 10

    def load_data(self):
        # open attributes, objects, qa, region descriptions and relationships
        ATTRIBUTE_PATH = self.DATA_DIR / Path(self.ATTRIBUTES[0]).name.replace(
            ".zip", ""
        )
        with open(ATTRIBUTE_PATH, "r") as f:
            # load json
            print("loading attributes")
            self.attributes = json.load(f)

        OBJECT_PATH = self.DATA_DIR / Path(self.OBJECTS[0]).name.replace(".zip", "")
        with open(OBJECT_PATH, "r") as f:
            # load json
            print("loading objects")
            self.objects = json.load(f)

        QA_PATH = self.DATA_DIR / Path(self.QUESTION_ANSWERS[0]).name.replace(
            ".zip", ""
        )
        with open(QA_PATH, "r") as f:
            # load json
            print("loading qa")
            self.qa = json.load(f)

        REGION_DESCRIPTION_PATH = self.DATA_DIR / Path(
            self.REGION_DESCRIPTIONS[0]
        ).name.replace(".zip", "")
        with open(REGION_DESCRIPTION_PATH, "r") as f:
            # load json
            print("loading region_descriptions")

            self.region_descriptions = json.load(f)

        RELATIONSHIP_PATH = self.DATA_DIR / Path(self.RELATIONSHIPS[0]).name.replace(
            ".zip", ""
        )
        with open(RELATIONSHIP_PATH, "r") as f:
            # load json
            print("loading relationships")
            self.relationships = json.load(f)

        self.IMAGE_IDS = []
        self.ID_TO_PATH_MAP = {}
        for image_dir in self.IMAGE_OUT_DIRS:
            print("loading image ids")
            paths = list(self.DATA_DIR.glob(f"{image_dir}/*.jpg"))
            ids = [int(Path(p).stem) for p in paths]
            for i, p in zip(ids, paths):
                self.ID_TO_PATH_MAP[i] = p

            self.IMAGE_IDS.extend(ids)

        IMAGE_METADATA_PATH = self.DATA_DIR / Path(
            self.IMAGE_METADATA_URLS[0]
        ).name.replace(".zip", "")
        with open(IMAGE_METADATA_PATH, "r") as f:
            print("loading image metadata")
            self.image_metadata = json.load(f)

        self.INDEX_TO_ID_MAP = {
            i: o.get("image_id") for i, o in enumerate(self.objects)
        }
        self.ID_TO_INDEX_MAP = {
            o.get("image_id"): i for i, o in enumerate(self.objects)
        }

    def download(self):
        # Download images if they don't exist
        IMAGE_TO_DL = [
            u
            for i, u in enumerate(self.IMAGE_URLS)
            if not (self.DATA_DIR / self.IMAGE_OUT_DIRS[i]).is_dir()
        ]
        download_mp(IMAGE_TO_DL, self.DATA_DIR)

        # unzip
        for url in IMAGE_TO_DL:
            unzip(self.DATA_DIR / Path(url).name, self.DATA_DIR)
            # remove zip file
            (self.DATA_DIR / Path(url).name).unlink()

        # Download everything else if it doesn't exist
        TO_DL = [
            u
            for u in self.ALL_URLS
            if not (self.DATA_DIR / Path(u).name.replace(".zip", "")).is_file()
        ]
        download_mp(TO_DL, self.DATA_DIR)

        # unzip
        for url in TO_DL:
            unzip(self.DATA_DIR / Path(url).name, self.DATA_DIR)
            # remove zip file
            (self.DATA_DIR / Path(url).name).unlink()

    @staticmethod
    def sort_by_size(regions, reverse=True):
        return sorted(
            regions,
            key=lambda d: d.get("width", 0) * d.get("height", 0),
            reverse=reverse,
        )

    def get_attributes(self, image_id):
        idx = self.ID_TO_INDEX_MAP.get(image_id)
        if idx is None:
            return
        return self.attributes[idx]

    def get_objects(self, image_id):
        idx = self.ID_TO_INDEX_MAP.get(image_id)
        if idx is None:
            return
        return self.objects[idx]

    def get_qa(self, image_id):
        idx = self.ID_TO_INDEX_MAP.get(image_id)
        if idx is None:
            return
        try:
            return self.qa[idx]
        except KeyError as e:
            print(e)
            import pdb

            pdb.set_trace()

    def get_region_description(self, image_id):
        idx = self.ID_TO_INDEX_MAP.get(image_id)
        if idx is None:
            return
        return self.region_descriptions[idx]

    def get_relationships(self, image_id):
        idx = self.ID_TO_INDEX_MAP.get(image_id)
        if idx is None:
            return
        return self.relationships[idx]

    def get_data(self, image_id, fields=None):
        FIELDS = [
            "attributes",
            "objects",
            "qa",
            "region_description",
            "relationships",
            "image_metadata",
        ]
        fns = {
            "attributes": self.get_attributes,
            "objects": self.get_objects,
            "qa": self.get_qa,
            "region_description": self.get_region_description,
            "relationships": self.get_relationships,
            "image_metadata": self.get_image_metadata,
        }
        data = {}
        if fields is None:
            fields = FIELDS
        for field in fields:
            assert field in FIELDS
            data[field] = fns[field](image_id)
        return data

    def get_image(self, image_id):
        path = self.ID_TO_PATH_MAP.get(image_id)
        if path is not None:
            return Image.open(path)

    def get_path(self, image_id):
        return self.ID_TO_PATH_MAP.get(image_id)

    def get_image_metadata(self, image_id):
        idx = self.ID_TO_INDEX_MAP.get(image_id)
        if idx is None:
            return
        return self.image_metadata[idx]

    @staticmethod
    def normalize(text):
        for punc in [",", ".", "!", ":", ";"]:
            text = text.replace(f" {punc}", f"{punc}")
        return text

    @staticmethod
    def get_location_descriptor(image_size, region_x_y_w_h):
        # image should be h, w
        image_h, image_w = image_size
        image_w_center = image_w // 2
        image_h_center = image_h // 2
        x, y, w, h = region_x_y_w_h
        if x < image_w_center and x + w < image_w_center:
            # left quadrant
            if (y < image_h_center) and (y + h < image_w_center):
                return "top left"
            elif (y < image_h_center) and (y + h > image_w_center):
                return "left"
            else:
                return "bottom left"
        elif x < image_w_center and x + w > image_w_center:
            # center quadrant
            if (y < image_h_center) and (y + h < image_w_center):
                return "top center"
            elif (y < image_h_center) and (y + h > image_w_center):
                return "center"
            else:
                return "bottom center"
        elif x > image_w_center:
            # right quadrant
            if (y < image_h_center) and (y + h < image_w_center):
                return "top right"
            elif (y < image_h_center) and (y + h > image_w_center):
                return "right"
            else:
                return "bottom right"
        raise Exception("shouldn't be here")

    def get_tags(self, image_id, max_n=5):
        attributes = self.get_attributes(image_id)["attributes"]
        unique_attributes = set()
        for attribute in attributes:
            name = random.choice(attribute["names"])
            a = attribute.get("attributes")
            if a:
                name = random.choice(a).strip() + " " + name
            unique_attributes.add(name)
        if max_n:
            unique_attributes = list(unique_attributes)[:max_n]
        return "Tags: " + ", ".join(list(unique_attributes))

    def get_caption(self, image_id, n=3, t=0.25, tag_prob=0.5):
        """
        Join together the top n largest region descriptions to make a caption.

        Each region description must take up at least t percent of the total image space to be included.
        """
        regions = self.get_region_description(image_id)
        if regions is None:
            print(f"No region description found for image id {image_id}")
            return
        regions = self.sort_by_size(regions["regions"])
        image_metadata = self.get_image_metadata(image_id)
        image_area = image_metadata.get("height", 0) * image_metadata.get("width", 0)
        caption = ""
        for i, d in enumerate(regions):
            if i >= n:
                break
            region_area = d.get("height", 0) * d.get("width", 0)
            if (
                image_area * t < region_area
            ) or i == 0:  # make sure there's always at least one caption
                caption += d["phrase"]
                if not caption.strip().endswith("."):
                    # add a full stop / separator if there isn't one already
                    caption += ". "
                else:
                    caption += " "
        caption = caption.strip()
        if random.random() < tag_prob:
            tags = self.get_tags(image_id)
        else:
            tags = ""
        if tags:
            if random.random() < 0.5:
                text = caption + ". " + tags
            else:
                if random.random() < 0.5:
                    caption = "Caption: " + caption
                text = tags + ". " + caption
        else:
            text = caption

        return self.normalize(text)

    def convert(self, out_dir, mode="copy"):

        # load data into memory
        self.load_data()

        count = 0
        n_folders = 0

        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        for image_id in tqdm(self.IMAGE_IDS, desc="converting visual genome dataset"):

            image_path = self.get_path(image_id)

            # get captions
            unique_captions = set()
            for _ in range(10):
                caption = self.get_caption(image_id)
                if caption is not None:
                    unique_captions.add(caption)

            qas = self.get_qa(image_id)
            if qas is not None:
                for q_a in qas.get("qas", []):
                    prompt = f"Q: {q_a['question']} A: {q_a['answer']}"
                    unique_captions.add(prompt)

            captions = list(unique_captions)

            # make out folders
            out_image_dir = out_dir / "images" / f"{n_folders:05}"
            out_image_dir.mkdir(exist_ok=True, parents=True)
            out_json_dir = out_dir / "image_data" / f"{n_folders:05}"
            out_json_dir.mkdir(exist_ok=True, parents=True)
            out_image_path = out_image_dir / Path(image_path).name
            out_json_path = out_json_dir / f"{Path(image_path).stem}.json"
            relative_im_path = str(out_image_path.relative_to(out_dir))
            # make data dict
            image_data = {
                "image_path": relative_im_path,
                "captions": captions,
                "metadata": {
                    "visual_genome_metadata": self.get_data(image_id),
                    "dataset": "visual_genome",
                },
            }

            # get image hash
            try:
                image = Image.open(image_path)
            except PIL.UnidentifiedImageError:
                print(f"Warning: Corrupted Image @ {image_path}")
                image = None
                image_data["metadata"]["corrupted"] = True
            if image is not None:
                image_data["metadata"]["image_hash"] = str(phash(image))

            # copy / move file
            if mode == "copy":
                thr = threading.Thread(
                    target=shutil.copyfile, args=(image_path, out_image_path)
                )
                thr.start()
            elif mode == "move":
                thr = threading.Thread(
                    target=shutil.move, args=(image_path, out_image_path)
                )
                thr.start()
            else:
                raise Exception("mode not recognized")

            # save json
            with open(out_json_path, "w") as f:
                json.dump(image_data, f, indent=4)

            # increment count
            count += 1
            if count % 10000 == 0:
                n_folders += 1


if __name__ == "__main__":
    dataset = VisualGenomeDownloader("/mnt/localdisk/visual_genome")
    dataset.download()
    dataset.convert(out_dir="/mnt/localdisk/visual_genome_converted")
