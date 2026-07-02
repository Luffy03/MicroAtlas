import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0030-sero-yap'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: MCF10A cells, spinning disk confocal microscopy')
print('  Original channels (C=4): Hoechst, Alexa488-YAP/TAZ, Alexa568-CD44, Alexa647-Phalloidin')
print('  Selected (2 of 4): Hoechst [0] + Alexa647-Phalloidin [3]')
print('  Removed: Alexa488-YAP/TAZ + Alexa568-CD44 (signaling markers, non-structural)')
print('  Input shape: (4, H, W) -> Output shape: (2, H, W)')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[0] == 4:
        img = img[[0, 3], :, :]
        tifffile.imwrite(file_path, img)
    elif len(img.shape) == 3 and img.shape[0] == 2:
        continue  # Already processed
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: Channel selection complete (kept Hoechst + Phalloidin).')
