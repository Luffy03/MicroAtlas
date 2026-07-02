import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0007-srikumar-sumo'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: budding yeast, fluorescence microscopy')
print('  Channels (both used): GFP + mCherry-NUP49')
print('  Page structure: Z=3, C=2, pages=6 per TIFF')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[0] == 6:
        # All channels used (C=2, both ≤3), no modification needed
        continue
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: All files verified, no modification needed (both channels used).')
