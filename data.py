from PIL import Image, ImageFont, ImageDraw
import numpy as np
import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms

EN_SET = 'qwertyuiopasdfghjklzxcvbnm'
EN_SET = EN_SET + ''.join([letter.upper() for letter in EN_SET])
EN_SET = list(EN_SET)

ZH_SET = [chr(i) for i in range(20000, 40000)]
ZH_SET = [chr(i) for i in range(12449, 12538)] # katakana

CWD = os.path.dirname(__file__)

FONT_LIST = glob.glob(os.path.join(CWD, 'fonts', '*.ttf'))
ZH_FONTS = [font for font in FONT_LIST if 'unicode' in font.lower()]

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

def get_random_char(char_set, font_list, size=(100,100)):
    W, H = size
    bg_color = (np.random.randint(100),)*3

    result = Image.new(size=size, mode='RGB', color=bg_color)
    
    text = np.random.choice(char_set, size=np.random.choice([1,2,3]))
    text = ''.join(text)
    text_color = tuple([np.random.randint(150,256) for _ in range(3)])
    font = ImageFont.truetype(np.random.choice(font_list), np.random.randint(W//2, 2*W//3))
    location = (np.random.randint(0,W//3), np.random.randint(0,H//3))

    draw = ImageDraw.Draw(result)
    draw.text(location, text, fill=text_color, font=font)

    return result


class CharDataset(Dataset):
    def __init__(self, en_set, zh_set, font_set, zh_fonts, transform):
        self.en_set = en_set
        self.zh_set = zh_set
        self.font_set = font_set
        self.zh_fonts = zh_fonts
        self.transform = transform

    def __len__(self):
        return 10000

    def __getitem__(self, x):
        image_en = get_random_char(self.en_set, self.font_set)
        image_zh = get_random_char(self.zh_set, self.zh_fonts)
        return {'en': self.transform(image_en), 'zh': self.transform(image_zh)}

dataset = CharDataset(EN_SET, ZH_SET, FONT_LIST, ZH_FONTS, transform)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    for x in loader:
        print(x['en'].size())
        print(x['zh'].size())
        break
