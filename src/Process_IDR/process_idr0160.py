import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0160-lippincott-pyroptosis'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: PBMCs and SH-SY5Y cells, modified Cell Painting assay')
print('  Channels selected (3 of 5): Hoechst + ConA-SYTO14 + Phalloidin-WGA')
print('  Removed: cleaved GSDMD + MitoTracker (pathway and organelle markers)')
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
print('  Channel selection (nuclei, ER, membrane only) handled downstream.')
