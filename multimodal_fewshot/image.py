import requests
from io import BytesIO
import  PIL.Image as PilImage

class ImageInput():
    """Wrapper to handle image inputs both from local paths and urls

    Args:
        path_or_url (str): path or link to image.
    """
    def __init__(self, path_or_url):
        
        self.path_or_url = path_or_url
        if self.path_or_url.startswith("http://") or self.path_or_url.startswith("https://"):
            try:
                response = requests.get(path_or_url)
                self.pil_image = PilImage.open(BytesIO(response.content))
            except:
                raise Exception(f'Could not retrieve image from url:\n{self.path_or_url}')
        else:
            self.pil_image = PilImage.open(path_or_url)

    def get_image(self):  ## to be called internally
        return self.pil_image