from cellpose import io, metrics, models, utils, transforms, dynamics
import time
from tqdm import trange
import torch
import numpy as np
from pathlib import Path
from natsort import natsorted
import argparse
import os
import sys
import logging
from datetime import datetime
from tqdm import tqdm
from cellpose.loader import CellposeDataset, BatchTransform, DataLoader, collate_skip_invalid
import torch.nn.functional as F
import torch.distributed as dist
from cellpose.distributed_utils import setup_distributed, cleanup_distributed, is_distributed, get_rank, get_world_size
import torch.multiprocessing as mp

torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda")


# ============================================================================
# Helper functions (migrated from microatlas_nips.py)
# ============================================================================

def extract_features_from_model(img_tensor, model, no_grad=True):
    """Extract features from model, returns [B, 256, H, W]
    Args:
        img_tensor: input image tensor
        model: model (may be wrapped in DDP)
        no_grad: whether to use torch.no_grad(), True for teacher model, False for student model
    """
    # Unwrap DDP or DataParallel model
    actual_model = model
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        actual_model = model.module
    
    # Get model dtype and convert input to same type
    model_dtype = getattr(actual_model, 'dtype', torch.float32)
    img_tensor = img_tensor.to(model_dtype)

    if no_grad:
        with torch.no_grad():
            x = actual_model.encoder.patch_embed(img_tensor)
            if actual_model.encoder.pos_embed is not None:
                x = x + actual_model.encoder.pos_embed
            for blk in actual_model.encoder.blocks:
                x = blk(x)
            features = actual_model.encoder.neck(x.permute(0, 3, 1, 2))
    else:
        x = actual_model.encoder.patch_embed(img_tensor)
        if actual_model.encoder.pos_embed is not None:
            x = x + actual_model.encoder.pos_embed
        for blk in actual_model.encoder.blocks:
            x = blk(x)
        features = actual_model.encoder.neck(x.permute(0, 3, 1, 2))
    return features


def compute_local_self_attention(feat, cosine=True):
    """
    For feature maps of shape [B, 256, H, W], splits into local patches along spatial dimensions,
    flattens each patch and computes pairwise similarity matrix between spatial positions.
    cosine=False: Gram matrix, F^T @ F (inner product)
    cosine=True:  Cosine similarity, L2 normalize each column first then F^T @ F, range [-1, 1]
    Returns shape: [B, n_patches, patch_size^2, patch_size^2]
    """
    B, C, H, W = feat.shape
    patch_h, patch_w = 8, 8
    n_h = H // patch_h
    n_w = W // patch_w
    n_patches = n_h * n_w
    patch_size = patch_h * patch_w

    attention_maps = []

    for b in range(B):
        feat_b = feat[b]  # [256, H, W]
        batch_attention = []
        for i in range(n_h):
            for j in range(n_w):
                # Take local patch: [256, 8, 8]
                patch = feat_b[:, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
                # Flatten spatial dimension: [256, 64]
                F_patch = patch.reshape(C, patch_size)
                if cosine:
                    # L2 normalize each column (spatial position)
                    F_patch = F_patch / (F_patch.norm(dim=0, keepdim=True) + 1e-8)
                # F^T @ F -> [64, 256] @ [256, 64] = [64, 64]
                attn = F_patch.t() @ F_patch
                batch_attention.append(attn)
        attention_maps.append(torch.stack(batch_attention, dim=0))

    # stack -> [B, n_patches, patch_size, patch_size]
    return torch.stack(attention_maps, dim=0)


def gram_loss(teacher_model, student_model, clean_img, noisy_img, upsample=False):
    """
    Compute Gram loss:
    - clean_img: [B, C, H, W]
      * upsample=True:  [B, 3, 512, 512] clean image, split into 4 patches (256,256) for processing (no_grad)
      * upsample=False: [B, 3, 256, 256] clean image, directly feed to teacher_model (no_grad)
    - noisy_img: [B, C, 256, 256] noisy image fed to student_model (requires grad)
      * upsample=True:  upsample to 512x512 then extract features
      * upsample=False: extract features directly
    Returns: scalar loss
    """
    if upsample:
        # Teacher: 512x512 -> split into 4 patches (256,256) -> extract features separately -> concatenate (no_grad=True)
        patch_tl = clean_img[:, :, :256, :256]
        patch_tr = clean_img[:, :, :256, 256:]
        patch_bl = clean_img[:, :, 256:, :256]
        patch_br = clean_img[:, :, 256:, 256:]

        feat_tl = extract_features_from_model(patch_tl, teacher_model, no_grad=True).float()
        feat_tr = extract_features_from_model(patch_tr, teacher_model, no_grad=True).float()
        feat_bl = extract_features_from_model(patch_bl, teacher_model, no_grad=True).float()
        feat_br = extract_features_from_model(patch_br, teacher_model, no_grad=True).float()

        top = torch.cat([feat_tl, feat_tr], dim=3)
        bot = torch.cat([feat_bl, feat_br], dim=3)
        teacher_feat = torch.cat([top, bot], dim=2)  # [B, 256, 64, 64]

        # Student: upsample noisy to 512x512 then extract features
        noisy_img_up = F.interpolate(noisy_img, size=(512, 512), mode='bilinear', align_corners=False)
        student_feat = extract_features_from_model(noisy_img_up, student_model, no_grad=False).float()  # [B, 256, 64, 64]
    else:
        # Teacher: 256x256 extract features directly -> [B, 256, 32, 32]
        teacher_feat = extract_features_from_model(clean_img, teacher_model, no_grad=True).float()
        # Student: 256x256 extract features directly -> [B, 256, 32, 32]
        student_feat = extract_features_from_model(noisy_img, student_model, no_grad=False).float()

    teacher_attention = compute_local_self_attention(teacher_feat, cosine=True)
    student_attention = compute_local_self_attention(student_feat, cosine=True)

    loss = torch.sqrt(torch.mean((teacher_attention - student_attention) ** 2))
    return loss


class TeeStream:
    """Write output to both original stream and file simultaneously"""
    def __init__(self, original_stream, file_handle):
        self.original_stream = original_stream
        self.file_handle = file_handle

    def write(self, data):
        self.original_stream.write(data)
        self.file_handle.write(data)
        self.file_handle.flush()

    def flush(self):
        self.original_stream.flush()
        self.file_handle.flush()

    def __getattr__(self, attr):
        return getattr(self.original_stream, attr)


# ============================================================================
# Data preparation functions (migrated from semi_train_IDR.py)
# ============================================================================

def prepare_file_lists(data_root, name='train'):
    """Prepare file lists (same as semi_train_IDR.py)"""
    data_root = Path(data_root)

    train_files = natsorted(data_root.glob(f"{name}/*.tif"))
    train_files = [f for f in train_files if "_mask" not in f.name and "_flow" not in f.name]

    # Filter out files that have corresponding mask and flows cache (to avoid recomputing flows during training)
    paired = [
        (f, f.parent / f"{f.stem}_mask.tif")
        for f in train_files
        if (f.parent / f"{f.stem}_mask.tif").exists()
        and (f.parent / f"{f.stem}_mask_flows.tif").exists()
    ]
    train_files = [p[0] for p in paired]
    train_mask_files = [p[1] for p in paired]

    return train_files, train_mask_files


def prepare_all_IDR_file_lists(image_root, mask_root, study_ls):
    """
    Batch prepare file lists for all IDR studies (one-time scan of all directories, avoiding repeated traversal)
    
    Compared to calling prepare_IDR_file_lists individually, this function scans image_root only once
    and mask_root only once, then uses dict+set for O(1) lookup, greatly reducing filesystem IO.

    Args:
        image_root: image root directory
        mask_root: mask root directory
        study_ls: list of study names

    Returns:
        dict: {study_name: (files, masks)}
    """
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    study_set = set(study_ls)

    # Step 1: one-time scan image_root, organize all image files by study
    study_files_dict = {}
    for folder in image_root.iterdir():
        if not folder.is_dir():
            continue
        folder_study = folder.name.split("-")[0]
        if folder_study not in study_set:
            continue
        tif_files = [f for f in folder.glob("*.tif") if "_mask" not in f.name and "_flow" not in f.name]
        study_files_dict.setdefault(folder_study, []).extend(tif_files)

    # Step 2: one-time scan mask_root, build "(folder_name, stem) -> valid" index
    # Only scanning _mask_flows.tif existence is enough (having flow cache implies having mask)
    valid_pairs = set()
    for folder in mask_root.iterdir():
        if not folder.is_dir():
            continue
        folder_name = folder.name
        for f in folder.glob("*_mask_flows.tif"):
            stem = f.stem.rsplit("_mask_flows", 1)[0]
            valid_pairs.add((folder_name, stem))

    # Step 3: pair based on index
    result = {}
    for study_name in study_ls:
        files = study_files_dict.get(study_name, [])
        paired_files = []
        paired_masks = []
        for img_file in files:
            folder_name = img_file.parent.name
            stem = img_file.stem
            if (folder_name, stem) in valid_pairs:
                paired_files.append(img_file)
                paired_masks.append(mask_root / folder_name / f"{stem}_mask.tif")
        result[study_name] = (paired_files, paired_masks)

    return result


def prepare_IDR_file_lists(image_root, mask_root, name):
    """Prepare IDR file lists (single study version, kept for backward compatibility)"""
    return prepare_all_IDR_file_lists(image_root, mask_root, [name]).get(name, ([], []))


# ============================================================================
# Core training function
# ============================================================================

def microatlas_IDR_train_seg(net, teacher_model, labeled_datasets, unlabeled_datasets,
                              test_files=None, test_mask_files=None,
                              batch_size=1, n_epochs=100,
                              learning_rate=1e-5, weight_decay=0.1, bsize=256,
                              save_path=None, models_dir=None, model_name=None, device=None,
                              num_workers=0, max_samples_per_epoch=1000, 
                              gram_weight=1.0, use_distributed=False, warmup_epochs=10,
                              save_every=10, add_noise=True, upsample=False,
                              dataset_weights=None, nimg_per_epoch=800,
                              crop_mode='center'):
    """
    Semi-supervised training function: perform supervised training (labeled data) first in each epoch,
    then unsupervised consistency training (unlabeled data) with Gram Loss.
    Supports per-epoch dynamic sampling and DDP distributed training.

    Args:
        net: network model (student)
        teacher_model: teacher model (fixed weights)
        labeled_datasets: dict of labeled datasets {name: (files, masks)}
        unlabeled_datasets: dict of unlabeled datasets {name: (files, masks)}
        test_files: list of test image file paths (optional)
        test_mask_files: list of test mask file paths (optional)
        batch_size: batch size
        n_epochs: number of training epochs
        learning_rate: learning rate
        weight_decay: weight decay
        bsize: crop size
        save_path: model save path
        models_dir: model save directory
        model_name: model name
        device: computing device
        num_workers: number of DataLoader workers
        max_samples_per_epoch: max samples per dataset per epoch
        gram_weight: Gram Loss weight
        use_distributed: whether to use DDP
        warmup_epochs: number of warmup epochs
        save_every: save model every N epochs

    Returns:
        tuple: (filename, train_losses, semi_losses, test_losses)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set correct device in DDP mode
    if use_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{rank}')
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
    else:
        rank = 0
        world_size = 1
    
    net = net.to(device)

    # If DDP or DP, get internal model to check dtype
    if use_distributed:
        actual_net = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
    else:
        actual_net = net.module if isinstance(net, torch.nn.DataParallel) else net
    
    if actual_net.dtype == torch.bfloat16:
        print(f"[Rank {rank}] Converting model from bfloat16 to float32")
        actual_net.dtype = torch.float32

    normalize_params = {"normalize": True, "do_3D": False}

    # ---- Learning rate schedule ----
    # Warmup phase: linearly increase from 0 to learning_rate
    LR = np.linspace(0, learning_rate, warmup_epochs)
    
    # Cosine decay phase: from warmup_epochs to n_epochs
    if n_epochs > warmup_epochs:
        decay_epochs = n_epochs - warmup_epochs
        # Use cosine decay to learning_rate * 0.01
        cosine_lr = learning_rate * 0.5 * (1 + np.cos(np.pi * np.arange(decay_epochs) / decay_epochs))
        LR = np.append(LR, cosine_lr)
    
    # Ensure learning rate array length equals n_epochs
    if len(LR) < n_epochs:
        LR = np.append(LR, learning_rate * 0.01 * np.ones(n_epochs - len(LR)))
    elif len(LR) > n_epochs:
        LR = LR[:n_epochs]

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # If using DDP, wrap model with DistributedDataParallel
    if use_distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )
        # Teacher model does not need DDP, but needs to be moved to the correct device
        teacher_model = teacher_model.to(device)
    
    teacher_model.eval()

    # ---- Loss function ----
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    filename = models_dir / model_name if models_dir is not None else None

    # Print only on rank 0
    if not use_distributed or dist.get_rank() == 0:
        n_labeled = sum(len(files) for files, _ in labeled_datasets.values())
        n_unlabeled = sum(len(files) for files, _ in unlabeled_datasets.values())
        print(f">>> n_epochs={n_epochs}, n_labeled={n_labeled}, n_unlabeled={n_unlabeled}")
        if test_files is not None:
            print(f">>> n_test={len(test_files)}")
        print(f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")
        print(f">>> gram_weight={gram_weight}, max_samples_per_epoch={max_samples_per_epoch}")
        print(f">>> warmup_epochs={warmup_epochs}")
        print(f">>> add_noise={add_noise}")
        print(f">>> upsample={upsample}")
        if filename is not None:
            print(f">>> saving model to {filename}")

    train_losses = np.zeros(n_epochs)
    semi_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    # Get model dtype for data conversion
    if use_distributed:
        actual_net_for_dtype = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
    else:
        actual_net_for_dtype = net.module if isinstance(net, torch.nn.DataParallel) else net
    model_dtype = actual_net_for_dtype.dtype

    # ---- CUDA Warmup: preheat critical kernels to avoid first-iteration compilation overhead ----
    if rank == 0:
        print("Running CUDA warmup...")
    
    with torch.no_grad():
        # Preheat student model forward
        warmup_img = torch.randn(1, 3, bsize, bsize, device=device, dtype=model_dtype)
        _ = net(warmup_img)
        
        # Preheat teacher model forward (including gram_loss)
        warmup_clean = torch.randn(1, 3, 512 if upsample else 256, 512 if upsample else 256, device=device, dtype=model_dtype)
        warmup_noisy = torch.randn(1, 3, 256, 256, device=device, dtype=model_dtype)
        _ = gram_loss(teacher_model, net, warmup_clean, warmup_noisy, upsample=upsample)
        
        # Preheat add_noise
        if add_noise:
            from cellpose.denoise import add_noise as _warmup_add_noise
            warmup_noise_in = torch.randn(1, 3, bsize, bsize, device=device, dtype=model_dtype)
            warmup_noise_in = torch.clamp(warmup_noise_in, 0.)
            _ = _warmup_add_noise(
                warmup_noise_in[:, :1], poisson=0.7, blur=0.7, ds_max=7, iso=True,
                downsample=0.0, beta=0.7, gblur=1.0,
                diams=torch.tensor([30.], device=device, dtype=torch.float32)
            )
    
    # Explicitly synchronize all GPUs
    if use_distributed:
        torch.cuda.synchronize()
        dist.barrier()
    
    if rank == 0:
        print("CUDA warmup done.")
    
    # ============ Training loop ============
    for iepoch in range(n_epochs):
        # ---- Per-epoch dynamic sampling ----
        if dataset_weights is not None:
            # === Paper strategy: probability-weighted sampling ===
            pool_files, pool_masks, pool_probs = [], [], []
            for dataset_name, (files, masks) in labeled_datasets.items():
                w = dataset_weights.get(dataset_name, 0)
                if w > 0 and len(files) > 0:
                    pool_files.extend(files)
                    pool_masks.extend(masks)
                    prob_per_image = w / len(files)
                    pool_probs.extend([prob_per_image] * len(files))
            pool_probs = np.array(pool_probs)
            pool_probs /= pool_probs.sum()
            n_sample = min(nimg_per_epoch * world_size, len(pool_files))
            indices = np.random.choice(len(pool_files), size=n_sample, replace=False, p=pool_probs)
            epoch_train_files = [pool_files[i] for i in indices]
            epoch_train_masks = [pool_masks[i] for i in indices]
        else:
            # === Original logic: equal uniform sampling (backward compatible) ===
            epoch_train_files, epoch_train_masks = [], []
            for dataset_name, (files, masks) in labeled_datasets.items():
                n_samples = min(len(files), max_samples_per_epoch)
                if n_samples > 0:
                    indices = np.random.permutation(len(files))[:n_samples]
                    epoch_train_files.extend([files[i] for i in indices])
                    epoch_train_masks.extend([masks[i] for i in indices])
        
        epoch_unlabeled_files, epoch_unlabeled_masks = [], []
        for dataset_name, (files, masks) in unlabeled_datasets.items():
            n_samples = min(len(files), max_samples_per_epoch)
            if n_samples > 0:
                indices = np.random.permutation(len(files))[:n_samples]
                epoch_unlabeled_files.extend([files[i] for i in indices])
                epoch_unlabeled_masks.extend([masks[i] for i in indices])
        
        nimg = len(epoch_train_files)
        n_unlabeled = len(epoch_unlabeled_files)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {iepoch}/{n_epochs}")
            print(f"Sampled: {nimg} labeled, {n_unlabeled} unlabeled")
            print(f"{'='*60}")
        
        # ---- Create DataLoader ----
        train_dataset = CellposeDataset(
            epoch_train_files, epoch_train_masks,
            transform=BatchTransform(bsize=bsize, crop_mode=crop_mode),
            normalize_params=normalize_params,
            device=device
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(train_dataset) if use_distributed else None,
            shuffle=not use_distributed,
            num_workers=num_workers,
            pin_memory=use_distributed,
            collate_fn=collate_skip_invalid,
            drop_last=True
        )
        
        unlabeled_loader = None
        if n_unlabeled > 0:
            unlabeled_dataset = CellposeDataset(
                epoch_unlabeled_files, epoch_unlabeled_masks,
                transform=BatchTransform(bsize=bsize, crop_mode=crop_mode),
                normalize_params=normalize_params,
                device=device
            )
            unlabeled_loader = DataLoader(
                unlabeled_dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.distributed.DistributedSampler(unlabeled_dataset) if use_distributed else None,
                shuffle=not use_distributed,
                num_workers=num_workers,
                pin_memory=use_distributed,
                collate_fn=collate_skip_invalid,
                drop_last=True
            )
        
        # ---- Test DataLoader ----
        test_loader = None
        if test_files is not None and len(test_files) > 0:
            test_dataset = CellposeDataset(
                test_files, test_mask_files,
                transform=BatchTransform(bsize=bsize, crop_mode=crop_mode),
                normalize_params=normalize_params,
                device=device
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False) if use_distributed else None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_distributed,
                collate_fn=collate_skip_invalid
            )

        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]
        
        # DDP needs to set sampler per epoch
        if use_distributed:
            train_loader.sampler.set_epoch(iepoch)
            if unlabeled_loader is not None:
                unlabeled_loader.sampler.set_epoch(iepoch)

        net.train()

        # ============ Phase 1: Supervised training ============
        lavg, nsum = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {iepoch} [Sup]", disable=use_distributed and rank != 0):
            if batch is None:
                continue
            if len(batch) == 2:
                images, masks = batch
                images = images.to(device).to(model_dtype)
                masks = masks.to(device).to(model_dtype)

                outputs = net(images)[0]

                veci = 5.0 * masks[:, -2:]
                loss_flow = mse_loss(outputs[:, -3:-1], veci) / 2.0
                loss_cell = bce_loss(outputs[:, -1], (masks[:, -3] > 0.5).to(outputs.dtype))
                loss = loss_flow + loss_cell

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = loss.item() * len(images)
                lavg += train_loss
                nsum += len(images)
                train_losses[iepoch] += train_loss

        # DDP mode: synchronize training loss
        if use_distributed:
            train_loss_tensor = torch.tensor([train_losses[iepoch]], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_losses[iepoch] = train_loss_tensor.item() / world_size
            nsum_tensor = torch.tensor([nsum], device=device)
            dist.all_reduce(nsum_tensor, op=dist.ReduceOp.SUM)
            nsum = nsum_tensor.item()

        train_losses[iepoch] /= max(nimg, 1)

        # Phase 1 complete: explicitly synchronize all GPUs, clear cache
        if use_distributed:
            torch.cuda.synchronize()
            dist.barrier()
        torch.cuda.empty_cache()
        
        # ============ Phase 2: Semi-supervised training (with Gram Loss) ============
        if unlabeled_loader is not None:
            nchan_noise = 1
            diam_mean = 30
            semi_nsum = 0
            semi_losses_breakdown = {"task": 0.0, "gram": 0.0}

            for batch in tqdm(unlabeled_loader, desc=f"Epoch {iepoch} [Semi]", disable=use_distributed and rank != 0):
                if batch is None:
                    continue
                images, masks = batch
                images = images.to(device).to(model_dtype)
                masks = masks.to(device).to(model_dtype)

                if add_noise:
                    from cellpose.denoise import add_noise
                    n_imgs = images.shape[0]
                    random_diam = diam_mean * (2 ** (2 * np.random.rand(n_imgs) - 1))

                    # Construct noisy images
                    images_noise = images.clone()
                    images_noise = torch.clamp(images_noise, 0.)
                    images_noise[:, :nchan_noise] = add_noise(
                        images_noise[:, :nchan_noise], poisson=0.7, blur=0.7, ds_max=7, iso=True,
                        downsample=0.0, beta=0.7, gblur=1.0,
                        diams=torch.from_numpy(random_diam).to(device).float()
                    )
                else:
                    images_noise = images.clone()
                    n_imgs = images.shape[0]

                # Noisy image prediction (backward pass)
                y_noisy = net(images_noise)[0]

                veci = 5.0 * masks[:, -2:]
                loss_flow = mse_loss(y_noisy[:, -3:-1], veci) / 2.0
                loss_cell = bce_loss(y_noisy[:, -1], (masks[:, -3] > 0.5).to(y_noisy.dtype))
                task_loss = loss_flow + loss_cell

                # ============ Gram Loss ============
                if upsample:
                    # upsample=True: teacher input is upsampled to 512x512
                    clean_img_for_gram = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
                else:
                    # upsample=False: teacher directly uses original 256x256
                    clean_img_for_gram = images

                gram_loss_value = gram_loss(
                    teacher_model=teacher_model,
                    student_model=net,
                    clean_img=clean_img_for_gram,
                    noisy_img=images_noise,
                    upsample=upsample
                )

                # Combine losses
                total_loss = task_loss + gram_weight * gram_loss_value

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Accumulate both losses separately
                semi_losses_breakdown["task"] += task_loss.item() * n_imgs
                semi_losses_breakdown["gram"] += gram_loss_value.item() * n_imgs
                semi_nsum += n_imgs

            # Compute average loss
            semi_losses[iepoch] = semi_losses_breakdown["task"] / max(semi_nsum, 1)
            gram_loss_avg = semi_losses_breakdown["gram"] / max(semi_nsum, 1)
            
            # DDP mode: synchronize semi training loss
            if use_distributed:
                semi_loss_tensor = torch.tensor([semi_losses[iepoch], gram_loss_avg], device=device)
                dist.all_reduce(semi_loss_tensor, op=dist.ReduceOp.SUM)
                semi_losses[iepoch] = semi_loss_tensor[0].item() / world_size
                gram_loss_avg = semi_loss_tensor[1].item() / world_size
            
            if rank == 0:
                print(f"Epoch {iepoch} [Semi] task_loss={semi_losses[iepoch]:.4f}, gram_loss={gram_loss_avg:.4f}")

        # ============ Test evaluation ============
        lavgt = 0
        test_samples = 0

        if test_loader is not None and (iepoch == 5 or iepoch % 10 == 0):
            net.eval()
            with torch.no_grad():
                for batch in test_loader:
                    if batch is None:
                        continue
                    images, masks = batch
                    images = images.to(device).to(model_dtype)
                    masks = masks.to(device).to(model_dtype)

                    outputs = net(images)[0]

                    loss_flow = mse_loss(outputs[:, -3:-1], masks[:, -2:]) * 5.0
                    loss_cell = bce_loss(outputs[:, -1], (masks[:, -3] > 0.5).to(outputs.dtype))
                    loss = loss_flow + loss_cell

                    test_loss = loss.item() * len(images)
                    lavgt += test_loss
                    test_samples += len(images)

            lavgt /= test_samples if test_samples > 0 else 1
            test_losses[iepoch] = lavgt

        lavg_epoch = lavg / nsum if nsum > 0 else 0
        # Print only on rank 0
        if not use_distributed or dist.get_rank() == 0:
            print(
                f"{iepoch}, train_loss={lavg_epoch:.4f}, semi_loss={semi_losses[iepoch]:.4f}, "
                f"test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time() - t0:.2f}s"
            )
        lavg, nsum = 0, 0

        # DDP mode: synchronize all processes at epoch end
        if use_distributed:
            dist.barrier()

        # ============ Save model (only on rank 0) ============
        if not use_distributed or dist.get_rank() == 0:
            # Check if save needed: every save_every epochs or the last epoch
            if (iepoch + 1) % save_every == 0 or iepoch == n_epochs - 1:
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
                print(f"saving network parameters to {filename0}")

                # Use net.module to get the original model when saving
                save_net = net.module if isinstance(net, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)) else net
                if hasattr(save_net, 'save_model'):
                    save_net.save_model(filename0)  # save_model does not add .pt extension
                else:
                    torch.save(save_net.state_dict(), str(filename0) + ".pt")
        
        # DDP mode: synchronize all processes after saving, prevent other ranks from exiting early
        if use_distributed:
            dist.barrier()

    # Final model has already been saved during the loop (if the last epoch meets save_every condition)
    # Verify file existence here (only on rank 0)
    if not use_distributed or dist.get_rank() == 0:
        # Check the model file for the last epoch
        final_filename = str(filename) + f"_epoch_{n_epochs-1:04d}"
        if os.path.exists(final_filename):
            file_size = os.path.getsize(final_filename)
            print(f"\n[Rank 0] Final model verified: {final_filename} ({file_size / 1024 / 1024:.2f} MB)")
        elif os.path.exists(final_filename + ".pt"):
            file_size = os.path.getsize(final_filename + ".pt")
            print(f"\n[Rank 0] Final model verified: {final_filename}.pt ({file_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"\n[Rank 0] WARNING: Final model NOT FOUND!")
    
    # DDP mode: synchronize all processes after final save
    if use_distributed:
        dist.barrier()

    return filename, train_losses, semi_losses, test_losses


# ============================================================================
# Main run function
# ============================================================================

def run(root, seed=0, ntrain=1,
        transformer=False, save_results=True,
        save_test_masks=False, keep_model=False,
        device=torch.device("cuda"), use_distributed=False, args=None, pretrained_model=None):
    """
    Main run function: integrates dataset preparation, model initialization and training call
    """
    # Set rank
    if use_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Ensure models directory exists (only on rank 0)
    models_dir = root / "models_microatlas"
    if rank == 0:
        models_dir.mkdir(exist_ok=True)

    # Logging file only created on rank 0
    log_file_handle = None
    log_filepath = None
    original_stdout = sys.stdout
    if rank == 0:
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        log_filepath = models_dir / log_filename
        log_file_handle = open(log_filepath, "w", encoding="utf-8")
        sys.stdout = TeeStream(original_stdout, log_file_handle)

    try:
        # Logging file only set on rank 0
        file_log_handler = None
        if rank == 0:
            io.logger_setup()
            # Add FileHandler after logger_setup, capture all logging output to log file
            file_log_handler = logging.FileHandler(log_filepath, encoding="utf-8")
            file_log_handler.setLevel(logging.INFO)
            file_log_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logging.getLogger().addHandler(file_log_handler)

            # Print and save all args parameters
            print("="*80)
            print("Training Configuration:")
            print("="*80)
            print(f"  root: {root}")
            print(f"  seed: {seed}")
            print(f"  ntrain: {ntrain}")
            print(f"  transformer: {transformer}")
            print(f"  save_results: {save_results}")
            print(f"  save_test_masks: {save_test_masks}")
            print(f"  keep_model: {keep_model}")
            print(f"  device: {device}")
            print("-"*80)
            if args is not None:
                print(f"  learning_rate: {args.learning_rate}")
                print(f"  weight_decay: {args.weight_decay}")
                print(f"  batch_size: {args.batch_size}")
                print(f"  n_epochs: {args.n_epochs}")
                print(f"  gram_weight: {args.gram_weight}")
                print(f"  max_samples_per_epoch: {args.max_samples_per_epoch}")
                print(f"  warmup_epochs: {args.warmup_epochs}")
                print(f"  save_every: {args.save_every}")
                print(f"  add_noise: {args.add_noise}")
                print(f"  upsample: {args.upsample}")
                print(f"  pretrained_model: {args.pretrained_model}")
                print(f"  multi_gpu: {args.multi_gpu}")
                print(f"  ddp: {args.ddp}")
            print("="*80)
            print()

            print(device)
            print(f"seed = {seed}, ntrain = {ntrain}")
        else:
            # Non-rank-0 processes also need basic logger_setup
            io.logger_setup()
            
        mtype = "sam" if transformer else "cyto3"
        netstr = f"microatlas"

        # ---- Load labeled training files (18 datasets) ----
        if rank == 0:
            print("Loading labeled datasets...")
        
        nips22_train_files, nips22_train_mask_files = prepare_file_lists('./data/nips22', name='train_labeled')
        Cellpose_train_files, Cellpose_train_mask_files = prepare_file_lists('./data/Cellpose')
        Deepbacs_train_files, Deepbacs_train_mask_files = prepare_file_lists('./data/Deepbacs')
        livecell_train_files, livecell_train_mask_files = prepare_file_lists('./data/livecell')
        Omnipose_fluro_train_files, Omnipose_fluro_train_mask_files = prepare_file_lists('./data/Omnipose_fluro')
        Omnipose_phc_train_files, Omnipose_phc_train_mask_files = prepare_file_lists('./data/Omnipose_phc')
        tissuenet_train_files, tissuenet_train_mask_files = prepare_file_lists('./data/tissuenet')
        YeaZ_train_files, YeaZ_train_mask_files = prepare_file_lists('./data/YeaZ')
        MoNuSAC_train_files, MoNuSAC_train_mask_files = prepare_file_lists('./data/MoNuSAC')
        MoNuSeg_train_files, MoNuSeg_train_mask_files = prepare_file_lists('./data/MoNuSeg')
        IHCTMA_train_files, IHCTMA_train_mask_files = prepare_file_lists('./data/IHCTMA')
        CoNIC_train_files, CoNIC_train_mask_files = prepare_file_lists('./data/CoNIC')
        CryNuSeg_train_files, CryNuSeg_train_mask_files = prepare_file_lists('./data/CryNuSeg')
        LynSec_train_files, LynSec_train_mask_files = prepare_file_lists('./data/LynSec')
        BCCD_train_files, BCCD_train_mask_files = prepare_file_lists('./data/BCCD')
        NuInSeg_train_files, NuInSeg_train_mask_files = prepare_file_lists('./data/NuInSeg')
        CPM_TNBC_train_files, CPM_TNBC_train_mask_files = prepare_file_lists('./data/CPM_TNBC')
        Pannuke_train_files, Pannuke_train_mask_files = prepare_file_lists('./data/Pannuke')

        # Group labeled data by dataset
        labeled_datasets = {
            'nips22': (nips22_train_files, nips22_train_mask_files),
            'Cellpose': (Cellpose_train_files, Cellpose_train_mask_files),
            'Deepbacs': (Deepbacs_train_files, Deepbacs_train_mask_files),
            'livecell': (livecell_train_files, livecell_train_mask_files),
            'Omnipose_fluro': (Omnipose_fluro_train_files, Omnipose_fluro_train_mask_files),
            'Omnipose_phc': (Omnipose_phc_train_files, Omnipose_phc_train_mask_files),
            'tissuenet': (tissuenet_train_files, tissuenet_train_mask_files),
            'YeaZ': (YeaZ_train_files, YeaZ_train_mask_files),
            'MoNuSAC': (MoNuSAC_train_files, MoNuSAC_train_mask_files),
            'MoNuSeg': (MoNuSeg_train_files, MoNuSeg_train_mask_files),
            'IHCTMA': (IHCTMA_train_files, IHCTMA_train_mask_files),
            'CoNIC': (CoNIC_train_files, CoNIC_train_mask_files),
            'CryNuSeg': (CryNuSeg_train_files, CryNuSeg_train_mask_files),
            'LynSec': (LynSec_train_files, LynSec_train_mask_files),
            'BCCD': (BCCD_train_files, BCCD_train_mask_files),
            'NuInSeg': (NuInSeg_train_files, NuInSeg_train_mask_files),
            'CPM_TNBC': (CPM_TNBC_train_files, CPM_TNBC_train_mask_files),
            'Pannuke': (Pannuke_train_files, Pannuke_train_mask_files),
        }

        # Paper dataset sampling weights (sum = 1.0)
        dataset_weights = {
            'Cellpose': 0.59,
            'livecell': 0.05,
            'tissuenet': 0.08,
            'YeaZ': 0.01,
            'Omnipose_fluro': 0.01,
            'Omnipose_phc': 0.02,
            'Deepbacs': 0.02,
            'nips22': 0.02,
            'MoNuSAC': 0.02,
            'MoNuSeg': 0.02,
            'IHCTMA': 0.02,
            'CoNIC': 0.02,
            'CryNuSeg': 0.02,
            'LynSec': 0.02,
            'BCCD': 0.02,
            'NuInSeg': 0.02,
            'CPM_TNBC': 0.02,
            'Pannuke': 0.02,
        }

        # Sample by ntrain (overall sampling)
        all_train_files = []
        all_train_mask_files = []
        for files, masks in labeled_datasets.values():
            all_train_files.extend(files)
            all_train_mask_files.extend(masks)
        
        ntrain = len(all_train_files) if ntrain == 0 else ntrain
        itrain = np.random.permutation(len(all_train_files))[:ntrain]
        # Note: do not reassign labeled_datasets here, keep original structure for per-epoch sampling

        if rank == 0:
            total_labeled = sum(len(files) for files, _ in labeled_datasets.values())
            print(f"Total labeled images: {total_labeled}")
            print(f"After ntrain sampling: {ntrain}")

        # ---- Load IDR unlabeled files ----
        if rank == 0:
            print("\nLoading IDR unlabeled datasets...")
        
        idr_image_root = './IDR_image'
        idr_mask_root = './IDR_mask'
        
        study_ls = ['idr0001',
                    'idr0002', 'idr0003',
                    'idr0006', 'idr0007', 'idr0008', 'idr0009', 'idr0010', 'idr0011',
                    'idr0012', 'idr0016', 'idr0017', 'idr0019', 'idr0020',
                    'idr0022', 'idr0025', 'idr0026',
                    'idr0028', 'idr0030', 'idr0033',
                    'idr0034', 'idr0035',
                    'idr0036', 'idr0037', 'idr0043',
                    'idr0056',
                    'idr0069', 'idr0071', 'idr0072',
                    'idr0076', 'idr0078', 'idr0080',
                    'idr0088', 'idr0090',
                    'idr0093', 'idr0112', 'idr0119', 'idr0120', 'idr0123', 'idr0128', 'idr0129', 'idr0133',
                    'idr0139', 'idr0143', 'idr0160']
        
        # One-time batch scan of all study file lists (avoid 44 repeated directory traversals)
        unlabeled_datasets = prepare_all_IDR_file_lists(idr_image_root, idr_mask_root, study_ls)
        if rank == 0:
            for study in study_ls:
                files, _ = unlabeled_datasets.get(study, ([], []))
                print(f"  {study}: {len(files)} unlabeled images")
        
        total_unlabeled = sum(len(files) for files, _ in unlabeled_datasets.values())
        if rank == 0:
            print(f"Total IDR unlabeled images: {total_unlabeled}")
            print(f"Number of IDR studies: {len(unlabeled_datasets)}")

        # ---- Initialize model (student supports checkpoint resume or train from scratch, teacher always loads pretrained weights) ----
        if pretrained_model:
            print(f"[Rank {rank}] Loading pretrained model from {pretrained_model}")
            model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)
        else:
            print(f"[Rank {rank}] No pretrained model specified, training from scratch")
            # Directly create Transformer network without loading any pretrained weights
            from cellpose.vit_sam import Transformer
            _student_net = Transformer(dtype=torch.bfloat16, bsize=256)
            model = type('_StudentModel', (), {'net': _student_net})()
        teacher_model_obj = models.CellposeModel(gpu=True)
        teacher_model = teacher_model_obj.net
        
        bsize = 256
        channels = None if transformer else [1, 0]

        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        batch_size = args.batch_size
        n_epochs = args.n_epochs
        gram_weight = args.gram_weight
        max_samples_per_epoch = args.max_samples_per_epoch

        if n_epochs > 0:
            tic = time.time()

            filename, train_losses, semi_losses, test_losses = microatlas_IDR_train_seg(
                net=model.net,
                teacher_model=teacher_model,
                labeled_datasets=labeled_datasets,
                unlabeled_datasets=unlabeled_datasets,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                bsize=bsize,
                save_path=root,
                models_dir=models_dir,  # All processes need models_dir
                model_name=netstr,
                num_workers=2,
                max_samples_per_epoch=max_samples_per_epoch,
                gram_weight=gram_weight,
                use_distributed=use_distributed,
                warmup_epochs=args.warmup_epochs,
                save_every=args.save_every,
                add_noise=bool(args.add_noise),
                upsample=bool(args.upsample),
                dataset_weights=dataset_weights,
                nimg_per_epoch=800,
                crop_mode=args.crop
            )

            toc = time.time() - tic
            if rank == 0:
                print(f"\nTotal training time: {toc:.2f}s")

        else:
            toc = 0
            filename, train_losses, semi_losses, test_losses = None, None, None, None

        # ---- Evaluation (only on rank 0) ----
        if rank == 0:
            torch.cuda.empty_cache()
                    
            # Reload the trained model
            # save_model saves files without .pt extension
            if filename is not None:
                model_path_no_ext = str(filename)
                model_path_with_ext = str(filename) + ".pt"
                
                if os.path.exists(model_path_no_ext):
                    print(f"Loading trained model from {model_path_no_ext}")
                    model = models.CellposeModel(gpu=True, pretrained_model=model_path_no_ext)
                elif os.path.exists(model_path_with_ext):
                    print(f"Loading trained model from {model_path_with_ext}")
                    model = models.CellposeModel(gpu=True, pretrained_model=model_path_with_ext)
                else:
                    print(f"WARNING: No trained model found at {filename}")
                    
            net = model.net
            # If DDP or DataParallel, get internal model
            if isinstance(net, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
                net = net.module
            net.eval()
            diameter = net.diam_labels.item() if not transformer else 30.

            # Load test data
            test_files = natsorted([
                tf for tf in (root / "test").glob("*.tif")
                if "_mask" not in str(tf) and "_flow" not in str(tf)
            ])

            print(f"nimg_test = {len(test_files)}")

            test_data = []
            print("loading test images")
            for i in trange(len(test_files)):
                img = io.imread(test_files[i])
                if len(img.shape) == 2:
                    img = np.tile(img[np.newaxis, :, :], (3, 1, 1))
                    img[1:] = 0
                test_data.append(img)

            test_masks = [io.imread(str(test_files[i])[:-4] + f'_mask.tif') for i in trange(len(test_files))]
            masks_pred = model.eval(test_data, diameter=diameter, channels=channels, niter=1000,
                                    batch_size=64, bsize=bsize)[0]

            # Save predicted masks as PNG images
            from PIL import Image

            save_dir = root / "predicted_masks"
            save_dir.mkdir(exist_ok=True)

            for i in range(len(masks_pred)):
                if len(test_files) > 0:
                    test_file_path = Path(test_files[i])
                    base_name = test_file_path.stem
                else:
                    base_name = f"test_{i:03d}"

                save_img_path = save_dir / f"{base_name}_{netstr}_pred.png"

                mask_img = masks_pred[i].astype(np.uint8)
                if mask_img.max() > 0:
                    mask_img = (mask_img / mask_img.max() * 255).astype(np.uint8)

                Image.fromarray(mask_img).save(save_img_path)
                print(f"Saved predicted mask: {save_img_path}")

            masks_gt = [tl.astype("uint16") for tl in test_masks]
            threshold = np.arange(0.5, 1.0, 0.05)
            ap, tp, fp, fn = metrics.average_precision(masks_gt, masks_pred, threshold=threshold)
            epsilon = 1e-10
            for n in range(len(tp)):
                denominator = tp[n] + fp[n] + fn[n] + epsilon
                ap[n] = tp[n] / denominator

            print(ap[:, [0, 5, 8]].mean(axis=0))

            if save_results:
                np.save(f"{models_dir}/{netstr}_AP_TP_FP_FN.npy", {
                    "threshold": threshold, "ap": ap, "tp": tp, "fp": fp, "fn": fn,
                    "ntrain_masks": ntrain, "test_files": test_files,
                    "diam_labels": diameter, "model_path": filename,
                    "train_losses": train_losses,
                    "semi_losses": semi_losses,
                    "test_losses": test_losses,
                    "test_masks_pred": masks_pred if save_test_masks else None,
                    "train_time": toc})

    finally:
        # Restore stdout and close log file (only on rank 0)
        if rank == 0:
            sys.stdout = original_stdout
            if file_log_handler is not None:
                logging.getLogger().removeHandler(file_log_handler)
                file_log_handler.close()
            if log_file_handle is not None:
                log_file_handle.close()
                print(f"Log saved to: {log_filepath}")
        
        # Need to synchronize all processes in DDP mode
        if use_distributed:
            dist.barrier()


# ============================================================================
# DDP launch logic
# ============================================================================

def ddp_main(rank, world_size, args):
    """Main function for DDP mode"""
    # Initialize process group
    setup_distributed(rank, world_size, 
                     master_addr=args.master_addr, 
                     master_port=str(args.master_port))
    
    try:
        # Call run function, passing args
        run(
            root=Path(args.root),
            ntrain=args.ntrain,
            seed=args.seed,
            transformer=args.transformer,
            use_distributed=True,
            args=args,
            pretrained_model=args.pretrained_model
        )
    finally:
        cleanup_distributed()


def run_ddp(args):
    """Launch DDP training"""
    world_size = torch.cuda.device_count()
    print(f"Launch DDP training, using {world_size} GPUs")
    print(f"Master: {args.master_addr}:{args.master_port}")
    
    # Launch using torch.multiprocessing.spawn
    mp.spawn(
        ddp_main,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


# ============================================================================
# Main entry
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./microatlas")
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transformer", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--multi_gpu", type=int, default=0, help="Use multi-GPU training (0 or 1)")
    parser.add_argument("--ddp", type=int, default=1, help="Use DDP mode (1) or DataParallel (0)")
    parser.add_argument("--master_addr", type=str, default="localhost", help="DDP master node address")
    parser.add_argument("--master_port", type=int, default=20123, help="DDP master node port")
    parser.add_argument("--gram_weight", type=float, default=1.0, help="Gram Loss weight")
    parser.add_argument("--max_samples_per_epoch", type=int, default=1000, help="Max samples per dataset per epoch")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of warmup epochs")
    parser.add_argument("--save_every", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--add_noise", type=int, default=1, help="Add noise augmentation to unlabeled data (0 or 1)")
    parser.add_argument("--upsample", type=int, default=0, help="In Gram Loss, upsample 256 noisy image to 512 before computing (0 or 1)")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Checkpoint resume: pretrained model path, e.g. ./microatlas/models_semi/semi_epoch_0164")
    parser.add_argument("--crop", type=str, default='center', choices=['random', 'center'],
                        help="Crop mode: random=random crop, center=crop centered on foreground")
    parser.add_argument("--keep_model", type=int, default=1, help="Keep model files (0=delete, 1=keep)")

    args = parser.parse_args()
    ntrain = args.ntrain
    seed = args.seed
    transformer = args.transformer
    root = Path(args.root)
    
    # Print all parameters
    print("="*50)
    print("Training Parameters:")
    print(f"  root: {args.root}")
    print(f"  ntrain: {args.ntrain}")
    print(f"  seed: {args.seed}")
    print(f"  transformer: {args.transformer}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  n_epochs: {args.n_epochs}")
    print(f"  multi_gpu: {args.multi_gpu}")
    print(f"  ddp: {args.ddp}")
    print(f"  master_addr: {args.master_addr}")
    print(f"  master_port: {args.master_port}")
    print(f"  gram_weight: {args.gram_weight}")
    print(f"  max_samples_per_epoch: {args.max_samples_per_epoch}")
    print(f"  warmup_epochs: {args.warmup_epochs}")
    print(f"  save_every: {args.save_every}")
    print(f"  add_noise: {args.add_noise}")
    print(f"  upsample: {args.upsample}")
    print(f"  pretrained_model: {args.pretrained_model}")
    print("="*50)
    
    # Choose training mode based on parameters
    if args.multi_gpu and args.ddp:
        # DDP mode (recommended)
        run_ddp(args)
    elif args.multi_gpu and not args.ddp:
        # DataParallel mode (alternative, not recommended)
        print("Warning: DataParallel mode has poor performance, DDP is recommended (--ddp 1)")
        run(Path(args.root), ntrain=args.ntrain, seed=args.seed, 
            transformer=args.transformer, use_distributed=False, args=args,
            pretrained_model=args.pretrained_model)
    else:
        # Single GPU mode
        run(Path(args.root), ntrain=args.ntrain, seed=args.seed,
            transformer=args.transformer, use_distributed=False, args=args,
            pretrained_model=args.pretrained_model)
