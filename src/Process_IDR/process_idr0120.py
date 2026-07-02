import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0120-german-immunologicalsynapse'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: Jurkat T cells and NK cells, spinning disk confocal microscopy')
print('  Channels selected (2 of 5): Hoechst (nuclei) + actin staining')
print('  Removed: LFA-1 and other immunological synapse markers')
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
print('  Channel selection (Hoechst and actin only) handled downstream.')
