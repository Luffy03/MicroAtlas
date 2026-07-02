import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0003-breker-plasticity'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: budding yeast, fluorescence microscopy')
print('  Channels (all 3 used): H2B-mCherry, GFP, bright-field')
print('  Note: each channel is stored as separate single-channel file')

count = 0
for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 2:
        count += 1
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: {count} single-channel files verified, no modification needed.')
