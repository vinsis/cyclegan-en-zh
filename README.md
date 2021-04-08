# cyclegan-en-zh
A CycleGAN model that converts the style of English letters to look like Mandarin or Katakana characters

A lot of code was taken from [aitorzip/PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN). Although I made it a lot more simpler (with a potential compromise in quality of results).

## Samples for English character to Katakana character mapping (after 20 epochs)
![](https://github.com/vinsis/cyclegan-en-zh/blob/main/results/en2jp_cropped.jpg)

## Samples for English character to Mandarin character mapping (after 10 epochs)
![](https://github.com/vinsis/cyclegan-en-zh/blob/main/results/en2zh_cropped.jpg)

## Samples for real image to sticker mapping (after 80 epochs)
![](https://github.com/vinsis/cyclegan-en-zh/blob/main/results/im2im_cropped.jpg)

More results can be seen [here](https://github.com/vinsis/cyclegan-en-zh/tree/main/results).

For both these datasets, I got the best results when [discrinimator weights](https://github.com/vinsis/cyclegan-en-zh/blob/main/main.py#L97-L98) were set to `0.1` (currently set to `0.3` in the code) and batch size was increased to 16. I couldn't go higher with the batch size due to memory issues but it's worth trying. 

## Other points
- Comment out [this line] to train on Mandarin characters.
- I also tried training on two image sets: real images and stickers. The files named `*im2im.py` contain the code for it. 
- You can run `python main.py` for training on character sets or `python main_im2im.py --/path/to/images1 --/path/to/images2` for training on image sets.
