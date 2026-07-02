import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0028-pascualvargas-rhogtpases'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: LM2 and MDA-MB-231 cells, spinning disk confocal microscopy')
print('  Original channels (C=4, channel-last): Hoechst, Alexa548-tubulin, Phalloidin488, Alexa647-YAP/TAZ')
print('  Selected (3 of 4): Hoechst [0] + Alexa548-tubulin [1] + Phalloidin488 [2]')
print('  Removed: Alexa647-YAP/TAZ (signaling marker)')
print('  Input shape: (H, W, 4) -> Output shape: (3, H, W)')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[-1] == 4:
        # Channel-last, select [0,1,2] and move to front
        img = np.moveaxis(img[..., [0, 1, 2]], -1, 0)
        tifffile.imwrite(file_path, img)
    elif len(img.shape) == 3 and img.shape[0] == 3:
        continue  # Already processed
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: Channel selection and reorientation complete.')
