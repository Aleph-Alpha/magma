import bz2
from pathlib import Path
from tqdm import tqdm
import csv
from PIL import Image
import math
from imagehash import phash
import json
from multiprocessing import Queue, Process, cpu_count
import time
from tor_requests import TorRequests
import sys
import re
import os
from dataset_utils import _download, resize, Counter, round_to_nearest


def _get_url_of_size(flickr_url, size):
    assert size in ["s", "q", "t", "m", "n", "w", "z", "c", "b", "o"]
    suffix = flickr_url.split("/")[-1]
    prefix = "/".join(flickr_url.split("/")[:-1])
    return f"{prefix}/{Path(suffix).stem}_{size}{Path(suffix).suffix}"


def get_original_url(flickr_url):
    return _get_url_of_size(flickr_url, "o")


def get_large_url(flickr_url):
    return _get_url_of_size(flickr_url, "b")


class YFCC100M:

    OPENAI_URL = (
        "https://openaipublic.azureedge.net/clip/data/yfcc100m_subset_data.tsv.bz2"
    )
    URL = "https://the-eye.eu/eleuther_staging/yfcc100m.csv"

    def __init__(self, data_dir, max_size=512, debug=False, tags_first_prob=0.3):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.max_size = max_size
        self.tags_first_prob = tags_first_prob

        out_path = self.data_dir / Path(self.URL).name
        if not out_path.exists():
            _download(self.URL, self.data_dir)

        # open csv
        csv.field_size_limit(
            sys.maxsize
        )  # some of the fields are very large, so we need to increase this
        self.data = csv.reader(open(out_path, "rt"))

    def _process_item(self, i, item):
        CAMERA_TITLE_FORMATS = [
            "img_",
            "_img",
            "HPI.?\d{4}",
            "PICT\d{4}",
            "P\d{7}",
            "^\d+$",
            "img\d{3,5}",
            "DSC.?\d{1,5}",
        ]
        if i == 0:
            return None  # headers

        url = item[14]
        title = item[6].replace("+", " ")
        description = item[7].replace("+", " ")
        user_tags = item[8].replace(",", ", ").replace("+", " ")
        machine_tags = item[9].replace(",", ", ").replace("+", " ")

        # get language_id for title and description
        title_lang_id = detect_lang(title)
        description_lang_id = detect_lang(description)

        # if the language isn't english, or can't be detected skip the image
        if title_lang_id != "en" and description_lang_id != "en":
            return None

        for (
            pattern
        ) in (
            CAMERA_TITLE_FORMATS
        ):  # if something like IMG_1234 is in the title, don't include it
            if re.search(pattern.lower(), title.lower()):
                title = ""
                break

        caption = ""
        if title:
            caption += title + "\n"
        if description:
            caption += description + "\n"

        tags = ""
        if user_tags:
            tags += user_tags
            tags += ", " if machine_tags else ""
        if machine_tags:
            tags += machine_tags
        if random.random() < self.tags_first_prob:
            caption = tags + "\n" + caption
        else:
            caption += tags
        caption = caption.strip()

        # decide and remove html tags
        caption = unquote(caption)
        caption = re.sub(CLEANR, "", caption)

        if caption:
            return url, caption

    def download_images(self, processes=None, use_tor=False):

        processes = processes or cpu_count()
        image_out_dir = self.data_dir / "images"
        image_out_dir.mkdir(exist_ok=True, parents=True)
        image_data_out_dir = self.data_dir / "image_data"
        image_data_out_dir.mkdir(exist_ok=True, parents=True)

        self.counter = Counter(print_every=1000)
        if use_tor:
            self.tor_req = TorRequests(processes)
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        def reader_proc(queue):
            ## Read from the queue; download image; compute hash + save image metadata
            if not use_tor:
                session = requests.Session()
            while True:
                d = queue.get()
                if d == "DONE":
                    print("received signal that queue is empty")
                    break
                idx, item = d
                processed = self._process_item(idx, item)

                if processed is None:
                    self.counter.increment(0)
                    continue  # filtered

                url, caption = processed
                # make subfolder if it doesn't exist
                subfolder_n = round_to_nearest(idx)
                image_subfolder = image_out_dir / f"{subfolder_n:09}"
                image_subfolder.mkdir(exist_ok=True, parents=True)
                image_data_subfolder = image_data_out_dir / f"{subfolder_n:09}"
                image_data_subfolder.mkdir(exist_ok=True, parents=True)

                image_out_path = image_subfolder / f"{idx:09}{Path(url).suffix}"
                image_data_out_path = image_data_subfolder / f"{idx:09}.json"
                image_data = {
                    "captions": [caption],
                    "metadata": {"url": url},
                    "image_path": str(image_out_path.relative_to(self.data_dir)),
                }
                if not image_out_path.is_file():

                    # do download
                    url = get_large_url(url)
                    image_data["large_url"] = url

                    def dl():
                        if use_tor:
                            retcode = self.tor_req.curl_dl(url, str(image_out_path))
                        else:
                            retcode = _download(
                                url,
                                None,
                                out_path=str(image_out_path),
                                disable_pbar=True,
                                session=session,
                            )
                        return retcode

                    try:
                        retcode = dl()
                    except requests.exceptions.ConnectionError:
                        print(f"Connection error for {url}... sleeping and retrying")
                        time.sleep(5)
                        try:
                            retcode = dl()
                        except Exception as e:
                            print(f"ERROR: {e}")
                            retcode = 0
                    except Exception as e:
                        print(f"ERROR: {e}")
                        retcode = 0

                    # resize image and compute image hash
                    if retcode:
                        try:
                            image = Image.open(image_out_path)
                            image = resize(image, self.max_size)
                            image_data["metadata"]["image_hash"] = str(phash(image))
                            image.save(str(image_out_path))
                        except Exception as e:
                            print(f"ERROR: {e}")
                            image_data["metadata"]["corrupted"] = True
                    else:
                        image_data["metadata"]["corrupted"] = True

                    # save json
                    with open(image_data_out_path, "w") as f:
                        json.dump(image_data, f, indent=4)

                else:
                    retcode = 1

                self.counter.increment(retcode)

        def writer(queue):
            ## Write to the queue
            for idx, item in enumerate(self.data):
                queue.put((idx, item))
            for _ in range(processes):
                queue.put("DONE")  # make sure each process gets the 'finish' signal
                queue.put("DONE")

        # then download hi-res images
        pqueue = Queue()  # writer() writes to pqueue from _this_ process

        writer_p = Process(target=writer, args=(pqueue,))
        writer_p.daemon = True
        writer_p.start()
        time.sleep(5)  # sleep to give the writer a chance to populate the queue

        reader_procs = []
        for _ in range(processes):
            reader_p = Process(target=reader_proc, args=(pqueue,))
            reader_p.daemon = True
            reader_p.start()  # Launch reader_proc() as a separate python process
            reader_procs.append(reader_p)

        # wait for all processes to finish
        for reader_p in reader_procs:
            reader_p.join()
        writer_p.join()


if __name__ == "__main__":
    dataset = YFCC100MDownloader("/mnt/localdisk/yfcc100m", openai_subset=True)
    dataset.download_images()