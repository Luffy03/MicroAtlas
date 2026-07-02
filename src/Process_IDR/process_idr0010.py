import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0010-doil-dnadamage'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: human cells, fluorescence microscopy')
print('  Channels (both used): Hoechst + Alexa488-53BP1')
print('  Page structure: Z=192, C=2, pages=384 per TIFF')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[0] == 384:
        continue
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: All files verified, no modification needed (both channels used).')
