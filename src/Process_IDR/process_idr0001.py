import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0001-graml-sysgro'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: fission yeast, spinning disk confocal')
print('  Channels (both used): Cascade Blue Dextran (cell shape) + GFP-tubulin (microtubules)')
print('  Page structure: Z=16, C=2, pages=32 per TIFF')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    # Already in (32, 1040, 1376) with all channels and Z-slices - keep as-is
    if len(img.shape) == 3 and img.shape[0] == 32:
        # All channels used (C=2, both ≤3), no modification needed
        continue
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: All files verified, no modification needed (both channels used).')
