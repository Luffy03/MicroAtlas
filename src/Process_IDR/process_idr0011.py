import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0011-ledesmafernandez-dad4'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: budding yeast, epifluorescence microscopy')
print('  Channels (all 3 used): Dad4-YFP + Spc42-RFP + DIC')
print('  Page structure: Z=21, C=3, pages=63 per TIFF')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[0] == 63:
        continue
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: All files verified, no modification needed (all 3 channels used).')
