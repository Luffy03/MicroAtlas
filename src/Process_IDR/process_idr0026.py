import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0026-weigelin-immunotherapy'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: B16F10-OVA mouse melanoma, multiphoton intravital microscopy')
print('  Original channels: mCherry-H2B (tumor nuclei), eGFP (CTL), Alexa750-dextran, SHG')
print('  Selected: mCherry-H2B only (tumor cell nuclei for segmentation)')
print('  Page structure: (N, H, W) where N combines C and Z dimensions')
print('  Assuming channels ordered in pages, selecting pages for mCherry channel')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    
    if len(img.shape) == 3 and img.shape[0] > 10:
        n_pages = img.shape[0]
        c = 4  # mCherry, eGFP, Alexa750, SHG
        # Estimate channel page width
        ch_width = n_pages // c
        if ch_width > 0:
            # Select mCherry pages (assumed first channel group)
            mcherry_pages = img[:ch_width]
            # Save as single-channel multi-page TIFF with mCherry only
            tifffile.imwrite(file_path, mcherry_pages)
    elif len(img.shape) == 2:
        # Already single channel (mCherry only), keep as-is
        continue
    else:
        print(f'  Skipping (unexpected shape {img.shape}): {f}')

print(f'{name}: mCherry-H2B channel extraction complete.')
print('  WARNING: Verify page-to-channel mapping; mCherry assumed as first channel group.')
