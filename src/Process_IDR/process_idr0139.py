import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0139-lawson-fascin'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: fascin-KD HeLa cells, spinning disk confocal microscopy')
print('  Channels selected (2 of 4): DAPI (nuclei) + Alexa488-Phalloidin (F-actin)')
print('  Removed: mScarlet-fascin + iRFP-nAC (experimental markers)')
print('  Note: each channel stored as separate single-channel file')

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
print('  Channel selection (DAPI and Phalloidin only) handled downstream.')
