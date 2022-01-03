from inference_api import caption

url = "https://s3-us-west-2.amazonaws.com/ai2-vision/aishwarya/mscoco_images/train2014/COCO_train2014_000000143482.jpg"
prompt = "Q:  When was this communication device invented? A:"
print(caption([url, prompt]))

prompt = ["https://www.h-hotels.com/_Resources/Persistent/b0a231fa7959f037b43c6e6583dcefd3898c4a2b/berlin-brandenburger-tor-04-2843x1600.jpg",
            "Q: What country is this? A: Germany",
            "https://storage.googleapis.com/afs-prod/media/media:003181861445403f903de279acae9914/3000.jpeg",
            "Q: What country is this? A: Tibet",
            "https://cdn.britannica.com/47/194547-050-52813FB0/aerial-view-Cairo-Egypt.jpg",
            "Q: What country is this? A: Egypt",
            "https://www.mexico-mio.de/fileadmin/_processed_/5/f/csm_teotihuacan-box_482dbc1216.jpg",
            "Q: What country is this? A:"]
print(caption(prompt))