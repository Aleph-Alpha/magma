from tqdm import tqdm
from dataset_utils import download_mp, _download, Counter, resize, round_to_nearest
from pathlib import Path
from multiprocessing import cpu_count, Pool, Process, Queue
import pandas as pd
import urllib.request
import json
import ftfy
import nltk
import re
from PIL import Image
from imagehash import phash
import os
import requests
import time


class LaionDownloader:

    URLS = [
        f"https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-{i:05}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
        for i in range(32)
    ]

    PROFANE_WORDS = set(["porn", "adult", "xxx", "sex", "fuck", "f*ck", "rape"])

    PHRASE_FILTER = [
        "load image into gallery viewer",
        "click to enlarge the picture",
        "profile photo",
        "embedded image permalink",
    ]

    URL_BLACKLIST = set(
        [
            i.strip()
            for i in urllib.request.urlopen(
                "https://raw.githubusercontent.com/chadmayfield/my-pihole-blocklists/master/lists/pi_blocklist_porn_all.list"
            )
            .read()
            .decode()
            .split("\n")
        ]
    )

    HTML_RE = re.compile("<.*?>")

    def __init__(
        self,
        out_path,
        processes=None,
        n_blocks=None,
        min_height=400,
        min_width=400,
        saved_images_max_size=512,
    ):
        self.out_path = Path(out_path)
        self.data_out_path = self.out_path / "image_data"
        self.images_out_path = self.out_path / "images"
        self.parquet_out_path = self.out_path / "parquets"
        self.min_height = min_height
        self.min_width = min_width
        self.saved_images_max_size = saved_images_max_size
        if processes is None:
            processes = cpu_count()
        self.processes = processes
        if n_blocks is not None:
            self.URLS = self.URLS[:n_blocks]

        # check if nltk 'punkt' is available, download if not
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        # check if nltk 'averaged_perceptron_tagger' is available, download if not
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("averaged_perceptron_tagger")

    def download(self):
        print("Downloading parquet files")
        download_mp(self.URLS, self.parquet_out_path)

        print("Downloading image data")
        self.counter = Counter(print_every=1000)
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        def reader_proc(queue):
            ## Read from the queue; download image; compute hash + save image metadata
            session = requests.Session()
            while True:
                try:
                    d = queue.get(timeout=60)
                except Exception as e:
                    print(e)
                    break
                if d == "DONE":
                    print("received signal that queue is empty")
                    break

                idx, item = d
                url, caption, metadata = item

                # make subfolder if it doesn't exist
                subfolder_n = round_to_nearest(idx)
                image_subfolder = self.images_out_path / f"{subfolder_n:09}"
                image_subfolder.mkdir(exist_ok=True, parents=True)
                image_data_subfolder = self.data_out_path / f"{subfolder_n:09}"
                image_data_subfolder.mkdir(exist_ok=True, parents=True)

                extension = ".jpg"
                image_out_path = image_subfolder / f"{idx:09}{extension}"
                image_data_out_path = image_data_subfolder / f"{idx:09}.json"
                image_data = {
                    "captions": [caption],
                    "metadata": metadata,
                    "image_path": str(image_out_path.relative_to(self.out_path)),
                }

                if not image_out_path.is_file():
                    try:
                        retcode = _download(
                            url,
                            out_dir=None,
                            out_path=str(image_out_path),
                            disable_pbar=True,
                            session=session,
                        )
                    except Exception as e:
                        print(f"ERROR: {e}")
                        retcode = 0

                    # resize image and compute image hash
                    if retcode:
                        try:
                            image = Image.open(image_out_path)
                            image = resize(image, self.saved_images_max_size)
                            image = image.convert("RGB")
                            image_data["metadata"]["image_hash"] = str(phash(image))
                            image.save(str(image_out_path))
                        except Exception as e:
                            print(f"PROCESSING ERROR: {e}")
                            image_data["metadata"]["corrupted"] = True
                    else:
                        image_data["metadata"]["corrupted"] = True

                    # save json
                    with open(image_data_out_path, "w") as f:
                        json.dump(image_data, f, indent=4)

                else:
                    retcode = 1

                self.counter.increment(retcode)

        def writer_proc(queue):
            ## Write to the queue
            for idx, item in enumerate(self.parquet_iterator()):
                queue.put((idx, item))
            for _ in range(self.processes):
                print("Writer process done iterating")
                queue.put("DONE")  # make sure each process gets the 'finish' signal
                queue.put("DONE")

        pqueue = Queue()  # writer() writes to pqueue from _this_ process

        writer_p = Process(target=writer_proc, args=(pqueue,))
        writer_p.daemon = True
        writer_p.start()
        print("Populating queue for 10 seconds...")
        time.sleep(10)  # sleep to give the writer a chance to populate the queue
        print("Done populating queue")

        reader_procs = []
        for _ in range(self.processes):
            reader_p = Process(target=reader_proc, args=(pqueue,))
            reader_p.daemon = True
            reader_p.start()  # Launch reader_proc() as a separate python process
            reader_procs.append(reader_p)

        # wait for all processes to finish
        for reader_p in reader_procs:
            reader_p.join()
        writer_p.join()

    def text_filter(
        self,
        text,
        url,
        repetition_cutoff=0.5,
        n_tokens_minimum=3,
        n_tokens_maximum=256,
        capitalized_cutoff=0.3,
    ):
        try:
            tokens_lower = nltk.word_tokenize(text.lower())
            parts_of_speech = nltk.pos_tag(tokens_lower)
            _, pos = zip(*parts_of_speech)
            tokens = nltk.word_tokenize(text)

            # discard if too few or too many tokens
            if len(tokens) <= n_tokens_minimum or len(tokens) >= n_tokens_maximum:
                return False

            # also discard if the ratio of capitalized words is too high
            capitalized_ratio = sum(
                [1 for i in tokens if i.isupper() or i.isdigit()]
            ) / len(tokens)
            if capitalized_ratio > capitalized_cutoff:
                return False

            # discard candidates with no determiner, no noun, or no preposition
            NOUN_POS = ["NN", "NNS", "NNP", "NNPS"]
            if "DT" not in pos:
                return False  # this rule doesn't seem great
            if all([i not in pos for i in ["NN", "NNS", "NNP", "NNPS"]]):
                return False

            # discard candidates with a high rate of repetition
            if len(set(tokens)) / len(tokens) < repetition_cutoff:
                return False

            # phrase filter
            for phrase in self.PHRASE_FILTER:
                if phrase in text:
                    return False

            return True
        except:
            return False

    def image_filter(self, url, height, width, aspect_ratio_cutoff=2.5):
        try:
            # filter by aspect ratio
            long_side = max(height, width)
            short_side = min(height, width)
            if long_side / short_side > aspect_ratio_cutoff:
                return False
            return True
        except:
            return False

    def url_filter(self, url):
        try:
            tld = Path(url).parts[1].lower().replace("www.", "")
        except:
            tld = url
        if tld in self.URL_BLACKLIST:
            return False
        for word in self.PROFANE_WORDS:
            if word in tld:
                return False
        return True

    def parquet_iterator(self):
        filtered_count = 0
        processed_count = 0
        self.filtered_ratio = None

        for parquet_file in self.parquet_out_path.glob("*.snappy.parquet"):
            # read into pandas
            print("Loading new parquet file")
            df = pd.read_parquet(parquet_file, engine="fastparquet")

            # filter out images that are too small
            larger_than_min_height = df["HEIGHT"] >= self.min_height
            larger_than_min_width = df["WIDTH"] >= self.min_width
            df = df[larger_than_min_height & larger_than_min_width]

            # filter out nsfw images (according to the dataset authors)
            nsfw_values = ["NSFW"]
            df = df[~df["NSFW"].isin(nsfw_values)]

            # iterate over the filtered dataframe
            for index, row in df.iterrows():
                # filter html tags / clean unicode
                text = ftfy.fix_text(re.sub(self.HTML_RE, "", row.TEXT))
                url = row.URL
                metadata = {"url": url, "height": row.HEIGHT, "width": row.WIDTH}
                if (
                    self.url_filter(url)
                    and self.text_filter(text, url)
                    and self.image_filter(url, row.HEIGHT, row.WIDTH)
                ):
                    yield url, text, metadata
                else:
                    filtered_count += 1
                processed_count += 1
                if processed_count % 1000 == 0:
                    self.filtered_ratio = filtered_count / processed_count


if __name__ == "__main__":
    dataset = LaionDownloader(out_path="/mnt/localdisk/laion", processes=256)
    dataset.download()
