from PIL import Image
import numpy as np
import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms

CWD = os.path.dirname(__file__)

transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.2),
                transforms.ToTensor()
            ])

class ImagePairDataset(Dataset):
    def __init__(self, dirA, dirB, transform=transform):
        self.imagesA = glob.glob(os.path.join(dirA, '*.jpg'))
        self.imagesB = glob.glob(os.path.join(dirB, '*.jpg'))
        print('Length of image sets A and B:', len(self.imagesA), len(self.imagesB))
        self.transform = transform
        self.SIZE = (100,100)

    def __len__(self):
        return max(len(self.imagesA), len(self.imagesB))

    def __getitem__(self, index):
        indexA = index % len(self.imagesA)
        indexB = index % len(self.imagesB)
        imageA = Image.open(self.imagesA[indexA]).convert('RGB').resize(self.SIZE)
        imageB = Image.open(self.imagesB[indexB]).convert('RGB').resize(self.SIZE)
        return {'A': self.transform(imageA), 'B': self.transform(imageB)}
