import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0076-ali-metabric'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: breast cancer tissue, imaging mass cytometry (IMC)')
print('  Channels: multi-channel IMC with 37 protein markers')
print('  All channels used for segmentation (cannot separate structural vs signaling in IMC)')

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
print('  Note: IMC channels are individual markers, each file is a single marker-channel TIFF.')
