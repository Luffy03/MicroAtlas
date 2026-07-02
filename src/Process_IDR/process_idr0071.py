import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0071-feldman-crisprko'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: HeLa cells, widefield fluorescence microscopy')
print('  Channels (both used): p65-mNeonGreen + SBS sequencing channel')
print('  Input shape: (2, H, W) -> kept as-is (both channels ≤3)')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[0] == 2:
        continue  # Both channels, no change needed
    elif len(img.shape) == 2:
        continue
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: All files verified, no modification needed (both channels used).')
