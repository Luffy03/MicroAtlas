from cellpose import io, metrics
import time
from tqdm import trange
import numpy as np
from pathlib import Path
from natsort import natsorted
import argparse
import os
import pandas as pd
import cv2
import torch

# Since torch_em and elf are installed, only mock vigra if needed
import sys

from micro_sam import util
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
)
MICROSAM_AVAILABLE = True

def convert_images_microsam(imgs0):
    imgs = []
    for img in imgs0:
        if img.ndim == 3:
            if np.array(img.shape).argmin() == 2:
                img = img.transpose(2, 0, 1)
            if np.ptp(img[1]) != 0 :
                img = img.astype("float32").mean(axis=0)
            else:
                img = img[0]
            
        img = np.stack((img,)*3, axis=-1)
        img = img.astype(np.float32)
        imgs.append(img)
    return imgs


def eval_microsam_on_datasets(test_root, save_results=True, device="cuda"):
    """
    Evaluate microsam performance on multiple datasets
    """
    if not MICROSAM_AVAILABLE:
        print("Error: micro_sam is not available.")
        print("Please install micro_sam with:")
        print("  conda install -c conda-forge microsam")
        return None, None
    
    io.logger_setup()
    print(f"Using device: {device}")
    
    # Initialize microsam model (same as benchmarks.py)
    model_type = "vit_l_lm"
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,
        checkpoint_path=None,
    )
    
    # Get all dataset folders
    test_root_ls = os.listdir(test_root)
    test_root_ls.sort()
    
    # Collect results for all datasets
    all_dataset_names = []
    all_image_names_dict = {}
    all_per_image_aps_dict = {}
    
    for dataset_name in test_root_ls:
        print(f"\n{'='*50}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*50}")
        
        root = os.path.join(test_root, dataset_name)
        root = Path(root)
        
        # Find test files
        test_files = list(root.glob("*.tif"))
        test_files = natsorted([tf for tf in test_files if "_mask" not in str(tf) and "_flows" not in str(tf)])
        
        if len(test_files) == 0:
            print(f"No test files found in {root}")
            continue
            
        print(f"Number of test images: {len(test_files)}")
        
        # Load images
        test_data = []
        print("Loading images...")
        for i in trange(len(test_files)):
            img = io.imread(test_files[i])
            test_data.append(img)
        
        # Load ground truth masks
        print("Loading ground truth masks...")
        test_masks = []
        for i in trange(len(test_files)):
            mask_path = str(test_files[i])[:-4] + '_mask.tif'
            if os.path.exists(mask_path):
                mask = io.imread(mask_path).astype("uint16")
                test_masks.append(mask)
            else:
                # Try alternative mask naming
                mask_path_alt = str(test_files[i]).replace(".tif", "_masks.tif")
                if os.path.exists(mask_path_alt):
                    mask = io.imread(mask_path_alt).astype("uint16")
                    test_masks.append(mask)
                else:
                    print(f"Warning: No mask found for {test_files[i]}")
                    test_masks.append(np.zeros(test_data[i].shape[:2], dtype="uint16"))
        
        # Convert images for microsam
        imgs = convert_images_microsam(test_data)
        
        # Run microsam prediction (same as benchmarks.py)
        masks_pred = []
        runtime = []
        
        print("Running microsam inference...")
        for i in trange(len(imgs)):
            tic = time.time()
            image = imgs[i]
            
            # Compute image embeddings
            image_embeddings = util.precompute_image_embeddings(
                predictor=predictor,
                input_=image,
                ndim=2,
                verbose=False,
            )
            
            # Initialize instance segmentation
            ais = InstanceSegmentationWithDecoder(predictor, decoder)
            ais.initialize(
                image=image,
                image_embeddings=image_embeddings,
            )
            
            # Generate predictions (output_mode='instance_segmentation' returns numpy array directly)
            prediction = ais.generate(output_mode='instance_segmentation')
            if prediction is not None and len(prediction) > 0:
                # with_background=True is handled by removing the largest component (background)
                # If needed, this is already done internally in newer versions
                masks_pred.append(prediction)
            else:
                masks_pred.append(np.zeros(image.shape[:2], "uint16"))
            
            toc = time.time() - tic
            runtime.append(toc)
        
        runtime = np.array(runtime)
        print(f"Average inference time: {runtime.mean():.3f}s per image")
        
        # Calculate metrics
        threshold = np.arange(0.5, 1.0, 0.05)
        ap, tp, fp, fn = metrics.average_precision(test_masks, masks_pred, threshold=threshold)
        
        # Calculate per-image AP
        image_names = []
        per_image_aps = []
        
        for i in range(len(test_files)):
            test_file_path = Path(test_files[i])
            base_name = test_file_path.stem
            image_names.append(base_name)
            
            # Calculate single image AP
            ap_i, tp_i, fp_i, fn_i = metrics.average_precision(
                [test_masks[i]], [masks_pred[i]], threshold=threshold
            )
            per_image_aps.append(ap_i[0])
        
        per_image_aps = np.array(per_image_aps)
        
        # Calculate overall AP
        ap_mean = ap[:, [0, 5, 8]].mean(axis=0)
        print(f"{dataset_name} AP@0.5, AP@0.75, AP@0.9: {ap_mean}")
        
        # Save results to txt file
        result_txt = '/data/linshan/Cellpose_data/all/eval_results_microsam.txt'
        with open(result_txt, 'a') as f:
            f.write(f"{dataset_name} {ap_mean}\n")
        
        # Save results
        all_dataset_names.append(dataset_name)
        all_image_names_dict[dataset_name] = image_names
        all_per_image_aps_dict[dataset_name] = per_image_aps
        
        # Save individual results
        if save_results:
            save_dir = f"./data/all/predicted_masks_per_dataset/{dataset_name}_microsam"
            os.makedirs(save_dir, exist_ok=True)
            
            from PIL import Image
            for i in range(len(masks_pred)):
                test_file_path = Path(test_files[i])
                base_name = test_file_path.stem
                save_path = os.path.join(save_dir, f"{base_name}_microsam_pred.png")
                
                mask_img = masks_pred[i].astype(np.uint8)
                if mask_img.max() > 0:
                    mask_img = (mask_img / mask_img.max() * 255).astype(np.uint8)
                
                Image.fromarray(mask_img).save(save_path)
    
    # Save overall results to Excel
    if save_results and len(all_dataset_names) > 0:
        print("\nSaving per-image AP results to Excel...")
        excel_path = f'./data/all/microsam_results.xlsx'
        
        # Find maximum number of images
        max_images = 0
        for dataset_name in all_dataset_names:
            if dataset_name in all_per_image_aps_dict:
                n_images = len(all_per_image_aps_dict[dataset_name])
                max_images = max(max_images, n_images)
        
        # Create worksheets for three thresholds
        threshold_indices = {0.5: 0, 0.75: 5, 0.9: 8}
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for thresh_name, thresh_idx in threshold_indices.items():
                data = {'Dataset': all_dataset_names}
                
                for img_idx in range(max_images):
                    col_name = str(img_idx + 1)
                    img_aps = []
                    for dataset_name in all_dataset_names:
                        if dataset_name in all_per_image_aps_dict:
                            per_image_aps = all_per_image_aps_dict[dataset_name]
                            if img_idx < len(per_image_aps):
                                ap_val = per_image_aps[img_idx, thresh_idx]
                            else:
                                ap_val = np.nan
                        else:
                            ap_val = np.nan
                        img_aps.append(ap_val)
                    
                    data[col_name] = img_aps
                
                df = pd.DataFrame(data)
                sheet_name = f'map{thresh_name}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Saved {sheet_name} sheet")
        
        print(f"Excel file saved to: {excel_path}")
    
    return all_dataset_names, all_per_image_aps_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, default='/data/linshan/Cellpose_data/all/test/')
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    test_root = Path(args.test_root)
    if not test_root.exists():
        print(f"Test root {test_root} does not exist!")
        exit(1)
    
    eval_microsam_on_datasets(
        test_root=test_root,
        save_results=args.save_results,
        device=args.device
    )