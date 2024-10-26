import os
import cv2
import torch
import albumentations as A

import config as CFG

class CLIPDataset(torch.utils.data.Dataset): 
    def __init__(self, img_filenames, text_prompts, tokenizer, transforms):
        """
        images and captions sizes should match
        """

        self.img_filenames = img_filenames
        self.text_prompts = list(text_prompts)
        self.encoded_prompts = tokenizer(list(text_prompts), padding=True,truncation=True, max_length= CFG.max_length)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_prompts.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.img_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.encoded_prompts[idx]

        return item
    
    def __len__(self):
        return len(self.text_prompts)


##FUNCTION FOR NORMALIZATION
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )