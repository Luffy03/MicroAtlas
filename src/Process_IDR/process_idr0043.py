import os
import numpy as np
from cellpose.io import imread
from tqdm import tqdm
import tifffile

root = './IDR_image'
name = 'idr0043-uhlen-humanproteinatlas'
path = os.path.join(root, name)

print(f'Processing {name}...')
print('  Dataset: human tissue sections, bright-field IHC')
print('  Channels (all 3 used): RGB (DAB + hematoxylin)')
print('  Reorienting: (H, W, 3) -> (3, H, W)')

for f in tqdm(os.listdir(path)):
    if not f.endswith('.tif'):
        continue
    file_path = os.path.join(path, f)
    img = imread(file_path)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = np.moveaxis(img, -1, 0)
        tifffile.imwrite(file_path, img)
    elif len(img.shape) == 3 and img.shape[0] == 3:
        continue  # Already processed
    else:
        print(f'  Unexpected shape {img.shape}: {f}')

print(f'{name}: Reorientation complete.')
