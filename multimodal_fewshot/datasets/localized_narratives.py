from dataset_utils import _download, load_jsonl, round_to_nearest
from pathlib import Path 
from tqdm import tqdm
from concurrent import futures
import sys 
import os 
import json

class LocalizedNarrativesDownloader:

    URL = 'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl'
    SUBSET = 'open_images'
    SPLIT = 'train'
    BUCKET_NAME = 'open-images-dataset'

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.image_data_dir = self.data_dir / 'image_data'

        self.image_data_dir.mkdir(exist_ok=True, parents=True)
        self.image_dir.mkdir(exist_ok=True, parents=True)

        _download(self.URL, self.data_dir)

        files = self.data_dir.glob('*.jsonl')
        self.data = []
        for file in files:
            self.data += load_jsonl(file)
        
        self.image_ids = [i.get('image_id') for i in self.data]
        pass
    
    def download_images(self):
        try:
            import boto3
            import botocore
        except ImportError:
            raise ImportError('Need boto3 installed to download OpenImages')

        bucket = boto3.resource(
        's3', config=botocore.config.Config(
        signature_version=botocore.UNSIGNED)).Bucket(self.BUCKET_NAME)

        image_folders = [str(self.image_dir / f"{round_to_nearest(i):09}") for i in range(len(self.image_ids))]
        for image_folder in list(set(image_folders)):
            Path(image_folder).mkdir(exist_ok=True, parents=True)

        data_folders = [str(self.image_data_dir / f"{round_to_nearest(i):09}") for i in range(len(self.image_ids))]
        for data_folder in list(set(data_folders)):
            Path(data_folder).mkdir(exist_ok=True, parents=True)

        def download_one_image(image_data, image_folder, data_folder):
            try:
                image_id = image_data.get('image_id')

                image_path = Path(os.path.join(image_folder, f'{image_id}.jpg'))
                image_data_path = Path(os.path.join(data_folder, f'{image_id}.json'))

                relative_path = image_path.relative_to(self.data_dir)

                bucket.download_file(f'{self.SPLIT}/{image_id}.jpg',
                                    str(image_path))
                
                image_data['image_path'] = str(relative_path)
                caption = image_data.pop('caption', '')
                image_data['captions'] = [caption]
                image_data['metadata'] = {'annotator_id': image_data.pop('annotator_id'), 'dataset_id': image_data.pop('dataset_id')}
                
                # save json
                with open(image_data_path, "w") as f:
                    json.dump(image_data, f, indent=4)

            except botocore.exceptions.ClientError as exception:
                sys.exit(
                    f'ERROR when downloading image `{self.SPLIT}/{image_id}`: {str(exception)}')
            except Exception as e:
                print(f'ERROR when downloading image `{self.SPLIT}/{image_id}`: {str(e)}')
        assert len(self.data) == len(image_folders)
        pbar = tqdm(desc='Downloading images', leave=True)
        with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            all_futures = [
                executor.submit(download_one_image, image_data, image_folder, data_folder) for (image_data, image_folder, data_folder) in zip(self.data, image_folders, data_folders)
            ]
            for future in futures.as_completed(all_futures):
                future.result()
                pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    dl = LocalizedNarrativesDownloader('/mnt/localdisk/localized_narratives')
    dl.download_images()