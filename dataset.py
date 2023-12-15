from PIL import Image
import os
import requests
from io import BytesIO

from torch.utils.data import Dataset

class COCO(Dataset):
    
    def __init__(self, data, images_path):
        self.data = data
        self.images_path = images_path
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.data.iloc[idx].filename)
        image = Image.open(image_path).convert("RGB")
        caption = self.data.iloc[idx].sentences
        return {'image' : image,
                'caption' : caption}
    
    def __len__(self):
        return len(self.data)
    
    def fetch_image(image_url):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img


def local_coco_collate_fn(batch):
    batch = [{'image' : obj['image'], 'caption' : obj['caption']} for obj in batch]
    return batch

def url_coco_collate_fn(batch):
    batch = [{'image' : COCO.fetch_image(obj['url']), 'caption' : obj['sentences']} for obj in batch]
    return batch
        
def collate_fn(data):
    return data