import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0123-mota-mifish'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: U2OS cells, multicolor iFISH')
print('  Original: Hoechst + up to 6 FISH probe channels in multi-page TIFF')
print('  Selected: Hoechst only (nuclear channel for segmentation)')
print('  Removed: all FISH probes (genomic locus markers)')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    
    if len(img.shape) == 3 and img.shape[0] > 10:
        n_pages = img.shape[0]
        # Assuming Hoechst is first channel group (contiguous pages)
        # Estimate channel groups: Hoechst + ~6 probe channels
        # For iFISH, Z-stack across channels
        # Assume 7 channel groups, keep only first group (Hoechst)
        c = 7  # Hoechst + 6 probe colors
        ch_width = n_pages // c
        if ch_width > 0:
            hoechst_pages = img[:ch_width]
            tifffile.imwrite(file_path, hoechst_pages)
    elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[0] <= 10):
        # Already single-channel or already processed
        continue
    else:
        print(f'  Skipping (unexpected shape {img.shape}): {f}')

print(f'{name}: Hoechst channel extraction complete.')
print('  WARNING: Verify page-to-channel mapping; Hoechst assumed as first channel group.')
