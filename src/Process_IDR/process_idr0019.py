import os
from cellpose.io import imread
from tqdm import tqdm

root = './IDR_image'
name = 'idr0019-sero-nfkappab'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: MCF10A cells, fluorescence microscopy')
print('  Channels (all 3 used): DAPI + Alexa488-p65 + DHE')

count = 0
for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[0] == 3:
        count += 1  # All 3 channels kept as-is
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: {count} files verified, no modification needed (all 3 channels used).')
