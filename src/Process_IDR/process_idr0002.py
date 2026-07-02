import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0002-heriche-condensation'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: HeLa cells (H2B-mCherry), epifluorescence time-lapse')
print('  Channels used: H2B-mCherry (chromatin, single channel)')

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
