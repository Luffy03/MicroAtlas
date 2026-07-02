import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0036-gustafsdottir-cellpainting'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: U2OS cells, Cell Painting assay (5 channels stored separately)')
print('  Channels selected (3 of 5): Hoechst + ConA-SYTO14 + Phalloidin-WGA')
print('  Note: each channel is a separate single-channel TIFF file')

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
print('  Channel selection (nuclei, ER, F-actin only) handled downstream.')
