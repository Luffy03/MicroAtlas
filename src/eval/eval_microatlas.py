from segment_anything import sam_model_registry
from cellpose import io, metrics, models, utils, train, transforms, dynamics
import time
from tqdm import trange
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from pathlib import Path
from natsort import natsorted
import argparse
import os
import pandas as pd

torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda")


def eval_subsets(root, dataset_name, seed=0, ntrain=1,
                 transformer=False, save_results=True,
                 save_test_masks=False, keep_model=False,
                 device=torch.device("cuda")):
    """
    Evaluate model performance on a single dataset
    Returns: (ap_mean, per_image_aps, image_names)
        - ap_mean: mean AP [AP@0.5, AP@0.75, AP@0.9]
        - per_image_aps: per-image AP array, shape (n_images, 10)
        - image_names: list of image names
    """
    io.logger_setup()
    print(device)
    print(f"seed = {seed}, ntrain = {ntrain}")
    mtype = "sam" if transformer else "cyto3"
    netstr = f"{mtype}_seed_{seed}_ntrain_{ntrain}"

    test_files = root.glob("*.tif")
    test_files = natsorted([tf for tf in test_files if "_mask" not in str(tf)])
    print(f"nimg_test = {len(test_files)}")

    if len(test_files) == 0:
        print(f"No test files found in {root}")
        return None, None, None

    test_data = []
    print("loading images")

    tf = test_files
    for i in trange(len(tf)):
        img = io.imread(tf[i])
        if len(img.shape) == 2:
            img = np.tile(img[np.newaxis, :, :], (3, 1, 1))
            img[1:] = 0
        test_data.append(img)

    print("loading labels and computing flows")
    test_masks = [io.imread(str(test_files[i])[:-4] + f'_mask.tif') for i in trange(len(test_files))]

    model = models.CellposeModel(gpu=True, pretrained_model='./microatlas/microatlas')

    bsize = 256 if transformer else 224
    channels = None if transformer else [1, 0]

    net = model.net
    net.eval()

    diameter = 30.
    masks_pred = model.eval(test_data, diameter=diameter, channels=channels, niter=1000,
                            batch_size=256, bsize=bsize)[0]

    # ========== Save predicted masks as PNG images ==========
    from PIL import Image

    if save_results:
        # Create save directory
        save_dir = f"./data/all/predicted_masks_per_dataset/{dataset_name}_microatlas"
        os.makedirs(save_dir, exist_ok=True)

        # Save predicted mask for each test image
        for i in range(len(masks_pred)):
            # Get original filename (without extension)
            if len(test_files) > 0:
                test_file_path = Path(test_files[i])
                base_name = test_file_path.stem
            else:
                base_name = f"test_{i:03d}"

            # Create save path
            save_path = os.path.join(save_dir, f"{base_name}_{netstr}_pred.png")

            # Convert numpy array to PIL image and save as PNG
            # Need to convert data type to uint8 or uint16 and ensure values are in 0-255 range
            mask_img = masks_pred[i].astype(np.uint8)
            if mask_img.max() > 0:
                mask_img = (mask_img / mask_img.max() * 255).astype(np.uint8)

            Image.fromarray(mask_img).save(save_path)
            print(f"Saved predicted mask: {save_path}")
    # ========== End of save predicted masks ==========

    masks_gt = [tl.astype("uint16") for tl in test_masks]
    threshold = np.arange(0.5, 1.0, 0.05)

    # Compute AP per image
    image_names = []
    per_image_aps = []

    for i in range(len(test_files)):
        test_file_path = Path(test_files[i])
        base_name = test_file_path.stem
        image_names.append(base_name)

        # Compute AP for a single image
        ap_i, tp_i, fp_i, fn_i = metrics.average_precision(
            [masks_gt[i]], [masks_pred[i]], threshold=threshold
        )
        epsilon = 1e-10
        for n in range(len(tp_i)):
            denominator = tp_i[n] + fp_i[n] + fn_i[n] + epsilon
            ap_i[n] = tp_i[n] / denominator

        per_image_aps.append(ap_i[0])  # ap_i shape is (1, 10), take first row

    per_image_aps = np.array(per_image_aps)  # shape (n_images, 10)

    # Compute overall AP
    ap, tp, fp, fn = metrics.average_precision(masks_gt, masks_pred, threshold=threshold)
    epsilon = 1e-10
    for n in range(len(tp)):
        denominator = tp[n] + fp[n] + fn[n] + epsilon
        ap[n] = tp[n] / denominator

    ap_mean = ap[:, [0, 5, 8]].mean(axis=0)
    print(dataset_name, ap_mean)

    # Save results to txt file
    result_txt = './data/all/eval_results_microatlas.txt'
    with open(result_txt, 'a') as f:
        f.write(f"{dataset_name} {ap_mean}\n")

    return ap_mean, per_image_aps, image_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transformer", type=int, default=1)
    parser.add_argument("--save_results", type=bool, default=True)

    args = parser.parse_args()
    ntrain = args.ntrain
    seed = args.seed
    transformer = args.transformer

    test_root = './data/all/test/'
    test_root_ls = os.listdir(test_root)
    test_root_ls.sort()

    # Collect per-image AP results for all datasets
    all_dataset_names = []
    all_image_names_dict = {}  # {dataset_name: [image_names]}
    all_per_image_aps_dict = {}  # {dataset_name: per_image_aps}

    for dataset_name in test_root_ls:

        root = os.path.join(test_root, dataset_name)
        root = Path(root)
        ap_mean, per_image_aps, image_names = eval_subsets(root, dataset_name=dataset_name, ntrain=ntrain, seed=seed,
                                                           transformer=args.transformer,
                                                           save_test_masks=True if seed == 0 else False,
                                                           keep_model=True)

        if ap_mean is not None:
            all_dataset_names.append(dataset_name)
            all_image_names_dict[dataset_name] = image_names
            all_per_image_aps_dict[dataset_name] = per_image_aps

    # Save per-image AP results to Excel
    if args.save_results and len(all_dataset_names) > 0:
        print("\nSaving per-image AP results to Excel...")
        mtype = "sam" if transformer else "cyto3"
        excel_path = f'./data/all/microatlas.xlsx'

        # Find maximum number of images
        max_images = 0
        for dataset_name in all_dataset_names:
            if dataset_name in all_per_image_aps_dict:
                n_images = len(all_per_image_aps_dict[dataset_name])
                max_images = max(max_images, n_images)

        # Create worksheets for three thresholds
        threshold_indices = {0.5: 0, 0.75: 5, 0.9: 8}  # indices in threshold array

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for thresh_name, thresh_idx in threshold_indices.items():
                # Create DataFrame
                # First column is dataset name, subsequent columns are image indices (1, 2, 3...)
                data = {'Dataset': all_dataset_names}

                # Create columns by image index
                for img_idx in range(max_images):
                    col_name = str(img_idx + 1)  # column names start from 1
                    img_aps = []
                    for dataset_name in all_dataset_names:
                        if dataset_name in all_per_image_aps_dict:
                            per_image_aps = all_per_image_aps_dict[dataset_name]
                            if img_idx < len(per_image_aps):
                                ap_val = per_image_aps[img_idx, thresh_idx]
                            else:
                                ap_val = np.nan  # this dataset does not have this image
                        else:
                            ap_val = np.nan
                        img_aps.append(ap_val)

                    data[col_name] = img_aps

                df = pd.DataFrame(data)

                # Write to worksheet, sheet names like 'map0.5', 'map0.75', 'map0.9'
                sheet_name = f'map{thresh_name}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Saved {sheet_name} sheet")

        print(f"Excel file saved to: {excel_path}")

