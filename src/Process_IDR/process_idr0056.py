import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0056-stojic-lncrnas'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: U2OS cells, fluorescence microscopy')
print('  Original channels (C=4): Hoechst, Alexa488-tubulin, Alexa568-Phalloidin, Alexa647-CEP215')
print('  Selected (3 of 4): Hoechst [0] + Alexa488-tubulin [1] + Alexa568-Phalloidin [2]')
print('  Removed: Alexa647-CEP215 (centrosome marker, non-structural)')
print('  Input shape: (4, H, W) -> Output shape: (3, H, W)')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[0] == 4:
        img = img[[0, 1, 2], :, :]
        tifffile.imwrite(file_path, img)
    elif len(img.shape) == 3 and img.shape[0] == 3:
        continue  # Already processed
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: Channel selection complete.')
