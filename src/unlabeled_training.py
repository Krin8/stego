"""Read training images without opening annotation files."""
from PIL import Image
import torch


class ImagesOnlyDataset:
    def __init__(self, dataset):
        if not dataset.use_preprocessed_data:
            raise ValueError('Images-only training requires approved preprocessed derivatives')
        self.dataset = dataset

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        with Image.open(self.dataset.samples[index][0]) as image:
            img = self.dataset.transform(image)
        # Compatibility placeholders contain no annotation-derived information.
        label = torch.full(img.shape[-2:], -1, dtype=torch.long)
        return img, label, torch.ones_like(label, dtype=torch.float32)
