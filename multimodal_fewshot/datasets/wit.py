from multiprocessing import Pool, cpu_count
from pathlib import Path
import gzip
import json
from PIL import Image
import os
import csv
import time
from multiprocessing import Process, Queue
import multiprocessing
import time
from tor_requests import TorRequests
from dataset_utils import _download, download_mp, resize, Counter
import re
import ftfy
from imagehash import phash
import shutil
from functools import partial
from tqdm import tqdm

def stream_tsv_gz(path):
    KEYS = None
    count = 0
    with gzip.open(path, "rt") as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        for line in tsv_reader:
            if count == 0:  # headers
                KEYS = line
                count += 1
            else:
                yield {k: v for k, v in zip(KEYS, line)}


def get_out_dir(base_out_dir, wiki_url):
    return Path(base_out_dir) / Path(*Path(wiki_url.split("://")[1]).parts[1:]).parent


def get_image_path(url, base_dir):
    # image path is the same path as url relative to some local directory
    path = url.replace("https://upload.wikimedia.org/", "")
    return Path(base_dir) / path


class WITDatasetDownloader:

    BASE_URL = "https://storage.googleapis.com/gresearch/wit/"
    LANGUAGE_PROMPTS = [
        "English: ",
        "Русский: ",
        "Deutsch: ",
        "Français: ",
        "Español: ",
        "Italiano: ",
        "Português: ",
        "Français : ",
        "Polski: ",
        "日本語: ",
    ]

    def __init__(self, out_dir, n=10, processes=None, image_max_size=512):
        # init variables
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.counter = Counter()
        self.URLS = [
            self.BASE_URL + f"wit_v1.train.all-{'%05d' % i}-of-00010.tsv.gz"
            for i in range(n)
        ]
        self.images_out_dir = self.out_dir / "images"
        self.data_out_dir = self.out_dir / "data"
        if processes is None:
            self.processes = cpu_count()
        self.image_max_size = image_max_size
        self.counter = None
        self.tor_req = None

    def download(self, use_tor=False, mirror=None):
        """
        Downloads the data
        """
        print("Downloading data")
        download_mp(self.URLS, self.data_out_dir)
        print("Downloading images")

        self.counter = Counter()
        if use_tor:
            self.tor_req = TorRequests(self.processes)

        # then download hi-res images
        pqueue = Queue()  # writer() writes to pqueue from _this_ process
        tsv_paths = list(self.data_out_dir.glob("*.tsv.gz"))

        writer_p = Process(
            target=self._writer_proc,
            args=(
                tsv_paths,
                pqueue,
            ),
        )
        writer_p.daemon = True
        writer_p.start()
        time.sleep(5)  # sleep to ensure writer populates queue

        reader_procs = []
        for p in range(self.processes):
            reader_p = Process(
                target=self._reader_proc,
                args=(pqueue, self.images_out_dir, mirror, use_tor),
            )
            reader_p.daemon = True
            reader_p.start()  # Launch reader_proc() as a separate python process
            reader_procs.append(reader_p)

        # wait for all processes to finish
        for reader_p in reader_procs:
            reader_p.join()
        writer_p.join()

    def _writer_proc(self, tsv_paths, queue):
        """
        Writes to the queue
        """
        ## Write to the queue
        for tsv_path in tsv_paths:
            for item in stream_tsv_gz(tsv_path):
                queue.put(item["image_url"])
        for _ in range(self.processes):
            queue.put("DONE")
            queue.put("DONE")

    def _reader_proc(self, queue, out_dir, mirror=None, use_tor=False):
        ## Read from the queue; this will be spawned as a separate Process
        while True:
            if queue.empty():
                break
            url = queue.get()
            if mirror is not None:
                url = url.replace(
                    "upload.wikimedia.org/",
                    mirror,
                )
            # make sure output dirs exist etc.
            final_out_dir = get_out_dir(out_dir, url)
            final_out_dir.mkdir(exist_ok=True, parents=True)
            out_filename = Path(url).name
            out_path = final_out_dir / out_filename
            if not os.path.isfile(out_path):
                # do download
                if use_tor:
                    retcode = self.tor_req.curl_dl(url, out_path)
                else:
                    retcode = _download(url, out_path, disable_pbar=True)
                # resize image
                if retcode:
                    try:
                        image = Image.open(out_path)
                        image = resize(image, self.image_max_size)
                        image.save(str(out_path))
                    except Exception as e:
                        print(f"ERROR: {e}")
            else:
                retcode = 1
            self.counter.increment(retcode)
            if url == "DONE":
                break

    def _process_item(self, item, idx, subfolder_n, languages, out_dir, mode):
        if item["language"] in languages:
            image_subfolder = out_dir / "images" / Path(f"{subfolder_n:09}")
            data_subfolder = out_dir / "image_data" / Path(f"{subfolder_n:09}")

            # make folders if they don't exist
            image_subfolder.mkdir(parents=True, exist_ok=True)
            data_subfolder.mkdir(parents=True, exist_ok=True)

            image_path = get_image_path(item["image_url"], self.images_out_dir)

            # save captions / metadata to json, and image to disk
            new_image_path = image_subfolder / f"{idx:09}{Path(image_path).suffix}"
            data_path = data_subfolder / f"{idx:09}.json"

            try:
                if new_image_path.exists():
                    print(f"Skipping {str(new_image_path)} as it already exists")
                    return
            except OSError as e:
                print(f"ERROR: {e}")
                return  # can get name too long errors

            captions = [self.get_caption(item)]
            try:
                image = Image.open(image_path)
                image_hash = str(phash(image))
                image_format = image.format
                corrupted = False
            except Exception as e:
                corrupted = True
                image_hash = None
                image_format = None
                print("ERROR: ", e)

            data = {
                "captions": captions,
                "image_path": str(new_image_path.relative_to(out_dir)),
                "metadata": {
                    "corrupted": corrupted,
                    "image_hash": image_hash,
                    "format": image_format,
                },
            }

            if mode == "move":
                try:
                    if image_path.exists():
                        image_path.rename(new_image_path)
                except OSError as e:
                    print(f"ERROR: {e}")
            elif mode == "copy":
                try:
                    if image_path.exists():
                        shutil.copy(image_path, new_image_path)
                except OSError as e:
                    print(f"ERROR: {e}")
            else:
                print(f"copying {image_path} to {new_image_path}")
                print(data)

            if mode != "dryrun":
                with open(data_path, "w") as f:
                    json.dump(data, f)

    def convert(self, out_dir, languages=None, mode="move"):
        if languages is None:
            languages = ["en"]
        assert mode in ["move", "copy", "dryrun"]
        out_dir = Path(out_dir)

        files = list(self.data_out_dir.glob("*.tsv.gz"))
        idx, subfolder_n = 0, 0
        processes = []
        pbar = tqdm(desc="converting WIT dataset")
        for f in files:
            for item in stream_tsv_gz(f):
                fn = partial(
                    self._process_item,
                    item=item,
                    idx=idx,
                    subfolder_n=subfolder_n,
                    languages=languages,
                    out_dir=out_dir,
                    mode=mode,
                )
                p = multiprocessing.Process(target=fn)
                p.start()
                processes.append(p)

                idx += 1
                if len(processes) >= self.processes:
                    # pop the oldest process and wait for it to finish
                    processes.pop(0).join()
                    pbar.update(1)

                if idx % 10000 == 0 and idx > 0:
                    print(f"Processed {idx} files")
                    subfolder_n += 1

    def get_caption(self, item):
        """
        Builds a text-string / prompt for a data sample.
        If the sample is the main image of the page, include page / context info in prompt, otherwise, just use caption info.
        """

        # remove double whitespaces in descriptions
        item["caption_reference_description"] = re.sub(
            r"  +", " ", item["caption_reference_description"]
        )
        item["caption_attribution_description"] = re.sub(
            r"  +", " ", item["caption_attribution_description"]
        )
        item["caption_alt_text_description"] = re.sub(
            r"  +", " ", item["caption_alt_text_description"]
        )
        item["context_page_description"] = re.sub(
            r"  +", " ", item["context_page_description"]
        )
        item["context_section_description"] = re.sub(
            r"  +", " ", item["context_section_description"]
        )

        # get all image captions
        captions = [
            item["caption_alt_text_description"],
        ]
        if not item["caption_reference_description"]:
            for language_prompt in self.LANGUAGE_PROMPTS:
                if language_prompt in item["caption_attribution_description"]:
                    # get the location of the language prompt
                    language_prompt_location = item[
                        "caption_attribution_description"
                    ].find(language_prompt)
                    if language_prompt_location > 0:
                        # truncate the description to the location of the language prompt
                        item["caption_attribution_description"] = item[
                            "caption_attribution_description"
                        ][:language_prompt_location]
                        break
                    else:
                        # just remove the prompt
                        item["caption_attribution_description"] = item[
                            "caption_attribution_description"
                        ].replace(language_prompt, "")

            captions = [
                item["caption_attribution_description"],
                item["caption_alt_text_description"],
            ]
        else:
            captions = [
                item["caption_reference_description"],
                item["caption_alt_text_description"],
            ]

        caption = "\n".join(list(set([i for i in captions if i])))

        # if main image, get titles, as well as all of the wiki page descriptions
        if item["is_main_image"] == "true":

            # get the most specific title possible
            if item["hierarchical_section_title"]:
                title = item["hierarchical_section_title"]
            elif item["section_title"]:
                title = item["section_title"]
            elif item["page_title"]:
                title = item["page_title"]
            else:
                title = ""

            # get the most specific description possible
            if item["context_section_description"]:
                description = item["context_section_description"]
            elif item["context_page_description"]:
                description = item["context_page_description"]
            else:
                description = ""

            caption += "\n" + title + "\n" + description

        return ftfy.fix_text(caption)




if __name__ == "__main__":
    dataset = WITDatasetDownloader("/mnt/shared_vol/wit")
    # dataset.download(use_tor=True)
    dataset.convert("/mnt/shared_vol/wit_converted", languages=["en"], mode="dryrun")
