import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0088-cox-phenomicprofiling'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: HepG2, A549, WPMY1, spinning disk confocal microscopy')
print('  Channels (all 3 used): BFP (nuclei) + GFP + RFP')

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
