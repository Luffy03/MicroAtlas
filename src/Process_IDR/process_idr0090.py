import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0090-ashdown-malaria'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: P. falciparum-infected RBCs, spinning disk confocal microscopy')
print('  Original channels (C=5): bright-field, DAPI, sfGFP, MitoTracker, WGA-Alexa633')
print('  Selected (3 of 5): bright-field [0] + DAPI [1] + WGA-Alexa633 [4]')
print('  Removed: sfGFP (parasite cytoplasm) + MitoTracker (mitochondria)')
print('  Page structure: Z=30, C=5 in TIFF (assuming C varies slowest in page order)')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    
    if len(img.shape) == 3 and img.shape[0] > 100:
        n_pages = img.shape[0]
        c = 5  # bright-field, DAPI, sfGFP, MitoTracker, WGA-633
        ch_width = n_pages // c
        if ch_width > 0:
            # Select pages for Ch0 (bright-field), Ch1 (DAPI), Ch4 (WGA)
            idx = list(range(0, ch_width)) + list(range(ch_width, 2 * ch_width)) + list(range(4 * ch_width, 5 * ch_width))
            img = img[idx]
            tifffile.imwrite(file_path, img)
    else:
        continue  # Already processed or single-channel

print(f'{name}: Channel selection complete (kept brightfield, DAPI, WGA).')
print('  WARNING: Verify page-to-channel mapping; assumes channel groups are contiguous in pages.')
