import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0072-schormann-subcellref'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: MCF10A cells, spinning disk confocal microscopy')
print('  Channels: single-channel (EGFP fusion proteins)')

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
