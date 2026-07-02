import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from cellpose import io, dynamics, transforms
import logging
from natsort import natsorted
import time
from tqdm import tqdm
import torch.distributed as dist
train_logger = logging.getLogger(__name__)

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

class CellposeDataset(Dataset):
    """Cellpose dataset class, loads images on demand"""

    def __init__(self, image_files, mask_files=None, transform=None,
                 normalize_params={"normalize": True}, device=None,
                 add_noise_files=None):
        """
        Args:
            image_files: list of image file paths
            mask_files: list of mask file paths (optional)
            transform: data augmentation transform
            normalize_params: normalization parameters
            device: computing device
            add_noise_files: set of image file paths that need noise added (optional), 50% probability of adding noise
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform
        self.normalize_params = normalize_params
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.add_noise_files = add_noise_files if add_noise_files is not None else set()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Load image on demand
            img = io.imread(self.image_files[idx])

            # Auto processing: unify to channels-first (C, H, W) format
            if len(img.shape) == 2:
                # Grayscale image
                img = np.stack([img, np.zeros_like(img), np.zeros_like(img)], axis=0)
            else:
                # Determine which dimension is the channel dimension (usually channel count is much smaller than height and width)
                # Find the smallest dimension as channel dimension
                channel_dim = np.argmin(img.shape)

                if channel_dim == 0:
                    # Already channels-first (C, H, W)
                    if img.shape[0] > 3:
                        # If more than 3 channels, take only first 3
                        img = img[:3]

                    elif img.shape[0] == 2:
                        # 2-channel expanded to 3-channel: add a blank channel
                        img = np.concatenate([img, np.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)], axis=0)

                    elif img.shape[0] == 1:
                        # Single channel duplicated to 3-channel
                        img = np.repeat(img, 3, axis=0)
                else:
                    # Channels-last (H, W, C) or (H, C, W), etc.
                    # Move channel dimension to front
                    img = np.moveaxis(img, channel_dim, 0)

                    # Ensure it is 3-channel
                    if img.shape[0] > 3:
                        img = img[:3]

                    elif img.shape[0] == 2:
                        # 2-channel expanded to 3-channel: add a blank channel
                        img = np.concatenate([img, np.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)], axis=0)

                    elif img.shape[0] == 1:
                        img = np.repeat(img, 3, axis=0)

            # Normalize
            if self.normalize_params.get("normalize", False):
                from cellpose.transforms import normalize_img
                img = normalize_img(img, normalize=self.normalize_params, axis=0)

            # Load mask (if available)
            mask = None
            if self.mask_files is not None:
                mask_file = self.mask_files[idx]
                mask = io.imread(mask_file)

                # Defensive check: ensure mask is a numpy array
                if not isinstance(mask, np.ndarray):
                    train_logger.warning(f"Sample {idx}: {self.image_files[idx]} - abnormal mask type {type(mask).__name__}, skipping")
                    return None
                
                if len(mask.shape) == 2:
                    # Check if corresponding flows cache file exists
                    mask_path = Path(mask_file)
                    flow_cache_path = mask_path.parent / f"{mask_path.stem}_flows.tif"
                            
                    # If cache file exists, load directly; otherwise compute and save
                    if flow_cache_path.exists():
                        train_logger.info(f"Loading cached flows from {flow_cache_path}")
                        mask = io.imread(flow_cache_path)
                    else:
                        train_logger.info(f"Computing flows for {mask_file} and caching to {flow_cache_path}")
                        # Compute flows (returns shape: [4, H, W], containing flows_y, flows_x, cellprob)
                        mask_computed = dynamics.labels_to_flows(mask[None, :, :], device=self.device)[0]
                        # io.imsave(str(flow_cache_path), mask_computed.astype(np.float32))
                        mask = mask_computed

                    # list of [4 x Ly x Lx] arrays: The flows for training the model. flows[k][0] is labels[k],
                    #         flows[k][1] is cell distance transform, flows[k][2] is Y flow, flows[k][3] is X flow,
                    #         and flows[k][4] is heat distribution.


            # Apply transforms
            if self.transform:
                if mask is not None:
                    img, mask = self.transform(img, mask)
                else:
                    img = self.transform(img)

            # Convert to tensor
            img = torch.from_numpy(img).float()

            # Add noise augmentation with 50% probability for specified datasets (e.g. all_test)
            if len(self.add_noise_files) > 0 and str(self.image_files[idx]) in self.add_noise_files:
                if np.random.rand() < 0.5:
                    from cellpose.denoise import add_noise
                    # add_noise expects input shape (nimg, nchan, Ly, Lx)
                    img_noise = torch.clamp(img.unsqueeze(0), 0.)  # (1, C, H, W)
                    diams = torch.tensor([30.], device=self.device, dtype=torch.float32)
                    # Only add noise to first channel (same as microatlas_train.py)
                    img_noise[:, :1] = add_noise(
                        img_noise[:, :1], poisson=0.7, blur=0.7, ds_max=7, iso=True,
                        downsample=0.0, beta=0.7, gblur=1.0,
                        diams=diams
                    )
                    img = img_noise.squeeze(0)

            if mask is not None:
                mask = torch.from_numpy(mask).float()
                return img, mask
            else:
                return img
        except Exception as e:
            train_logger.warning(f"Skipping error sample {idx}: {self.image_files[idx]} - {str(e)}")
            return None


class BatchTransform:
    """Batch transform for data augmentation"""

    def __init__(self, bsize=256, scale_range=0.5, rescale=None, crop_mode='center'):
        self.bsize = bsize
        self.scale_range = scale_range
        self.rescale = rescale
        self.crop_mode = crop_mode

    def __call__(self, img, mask=None):
        # Random rotation and resize
        from cellpose.transforms import random_rotate_and_resize

        center_override = None
        if mask is not None and self.crop_mode == 'center':
            # mask[0] = instance labels (0=background, >0=cell)
            fg_mask = (mask[0] if mask.ndim == 3 else mask) > 0
            if fg_mask.any():
                fg_coords = np.where(fg_mask)
                idx = np.random.randint(len(fg_coords[0]))
                # center_override = (cx, cy) = (column coordinate, row coordinate)
                center_override = (fg_coords[1][idx], fg_coords[0][idx])

        if mask is not None:
            img, mask = random_rotate_and_resize(
                [img], Y=[mask], rescale=self.rescale,
                scale_range=self.scale_range, xy=(self.bsize, self.bsize),
                center_override=center_override
            )[:2]
            return img[0], mask[0]
        else:
            img = random_rotate_and_resize(
                [img], Y=None, rescale=self.rescale,
                scale_range=self.scale_range, xy=(self.bsize, self.bsize)
            )[0]
            return img[0]


def collate_skip_invalid(batch):
    """Collate function that filters out invalid samples (returning None)"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    if isinstance(batch[0], tuple):
        images = torch.stack([item[0] for item in batch], dim=0)
        masks = torch.stack([item[1] for item in batch], dim=0)
        return images, masks
    else:
        return torch.stack([item for item in batch], dim=0)


def train_seg(net, train_files, train_mask_files=None, test_files=None,
                       test_mask_files=None, batch_size=1, n_epochs=100,
                       learning_rate=1e-5, weight_decay=0.1, bsize=256,
                       save_path=None, model_name=None, device=None,
                       num_workers=0, nimg_per_epoch=None, use_distributed=False,
                       max_samples_per_epoch=None, warmup_epochs=None, save_every=None,
                       rescale=None, scale_range=0.5,
                       labeled_datasets=None, dataset_weights=None,
                       use_cosine_decay=False,
                       models_dir=None, crop_mode='center',
                       add_noise_files=None):
    """
    Improved training function using DataLoader for streaming data loading
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

    # If model is bfloat16, convert to float32 to avoid type errors
    # If DDP or DP, get internal model to check dtype
    if use_distributed:
        actual_net = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
    else:
        actual_net = net.module if isinstance(net, torch.nn.DataParallel) else net
    
    if actual_net.dtype == torch.bfloat16:
        print(f"[Rank {rank}] Converting model from bfloat16 to float32")
        actual_net.dtype = torch.float32

    # Create dataset
    normalize_params = {"normalize": True, "do_3D": False}

    train_dataset = CellposeDataset(
        train_files, train_mask_files,
        transform=BatchTransform(bsize=bsize, scale_range=scale_range, crop_mode=crop_mode),
        normalize_params=normalize_params,
        device=device,
        add_noise_files=add_noise_files
    )

    nimg = len(train_dataset)
    nimg_per_epoch = nimg if nimg_per_epoch is None else min(nimg_per_epoch, nimg)

    # Create data loader
    if use_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_skip_invalid,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_skip_invalid,
            drop_last=True
        )

    # Test set
    test_dataset = None
    test_loader = None
    if test_files is not None:
        test_dataset = CellposeDataset(
            test_files, test_mask_files,
            transform=BatchTransform(bsize=bsize, scale_range=scale_range, crop_mode=crop_mode),
            normalize_params=normalize_params,
            device=device
        )
        if use_distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_skip_invalid
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                collate_fn=collate_skip_invalid
            )

    # Learning rate schedule (supports configurable warmup and cosine decay)
    warmup_len = min(warmup_epochs if warmup_epochs is not None and warmup_epochs > 0 else 10, n_epochs)
    if use_cosine_decay:
        LR = np.linspace(0, learning_rate, warmup_len)
        decay_epochs = n_epochs - warmup_len
        if decay_epochs > 0:
            cosine_lr = learning_rate * 0.5 * (1 + np.cos(np.pi * np.arange(decay_epochs) / decay_epochs))
            LR = np.append(LR, cosine_lr)
    else:
        LR = np.linspace(0, learning_rate, warmup_len)
        LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - warmup_len)))
        if n_epochs > 300:
            LR = LR[:-100]
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10))
        elif n_epochs > 99:
            LR = LR[:-50]
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    # If using DDP, wrap model with DistributedDataParallel
    if use_distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, 
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )

    # Loss functions
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # Print only on rank 0
    if not use_distributed or dist.get_rank() == 0:
        print(f">>> n_epochs={n_epochs}, n_train={nimg}")
        if test_dataset:
            print(f">>> n_test={len(test_dataset)}")
        print(f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    if models_dir is not None:
        models_dir = Path(models_dir)
        filename = models_dir / model_name
        if not use_distributed or dist.get_rank() == 0:
            models_dir.mkdir(exist_ok=True)
            print(f">>> saving model to {filename}")
    else:
        save_path = Path.cwd() if save_path is None else Path(save_path)
        filename = save_path / "models" / model_name
        if not use_distributed or dist.get_rank() == 0:
            (save_path / "models").mkdir(exist_ok=True)
            print(f">>> saving model to {filename}")

    # Training records
    lavg, nsum = 0, 0
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    # Get model dtype for data conversion
    if use_distributed:
        actual_net_for_dtype = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
    else:
        actual_net_for_dtype = net.module if isinstance(net, torch.nn.DataParallel) else net
    model_dtype = actual_net_for_dtype.dtype

    # ---- CUDA Warmup: preheat GPU kernels to avoid first-iteration compilation overhead ----
    if not use_distributed or dist.get_rank() == 0:
        print("Running CUDA warmup...")

    with torch.no_grad():
        warmup_img = torch.randn(1, 3, bsize, bsize, device=device, dtype=model_dtype)
        _ = net(warmup_img)

    if use_distributed:
        torch.cuda.synchronize()
        dist.barrier()

    # ---- Noise Warmup: preheat add_noise GPU kernel ----
    if add_noise_files is not None and len(add_noise_files) > 0:
        with torch.no_grad():
            from cellpose.denoise import add_noise as _warmup_add_noise
            warmup_noise_in = torch.randn(1, 3, bsize, bsize, device=device, dtype=model_dtype)
            warmup_noise_in = torch.clamp(warmup_noise_in, 0.)
            _ = _warmup_add_noise(
                warmup_noise_in[:, :1], poisson=0.7, blur=0.7, ds_max=7, iso=True,
                downsample=0.0, beta=0.7, gblur=1.0,
                diams=torch.tensor([30.], device=device, dtype=torch.float32)
            )

    if use_distributed:
        torch.cuda.synchronize()
        dist.barrier()

    if not use_distributed or dist.get_rank() == 0:
        print("CUDA warmup done.")

    # Save interval: when save_every is None, default to save every 10 epochs
    save_interval = save_every if save_every is not None and save_every > 0 else 10

    for iepoch in range(n_epochs):
        # Per-epoch dynamic sampling
        if labeled_datasets is not None:
            if dataset_weights is not None:
                # === Paper strategy: probability-weighted sampling (each image probability = w_d / N_d) ===
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

                # Total pool = 800 per GPU x num GPUs, ensures ~800 images per GPU
                n_sample = min(nimg_per_epoch * world_size, len(pool_files))
                indices = np.random.choice(len(pool_files), size=n_sample, replace=False, p=pool_probs)
                epoch_files = [pool_files[i] for i in indices]
                epoch_masks = [pool_masks[i] for i in indices]

                # Datasets with weight <= 0 are fully included without sampling (e.g. all_test)
                for dataset_name, (files, masks) in labeled_datasets.items():
                    w = dataset_weights.get(dataset_name, 0)
                    if w <= 0 and len(files) > 0:
                        if not use_distributed or dist.get_rank() == 0:
                            print(f"  >> Full-include: {dataset_name} ({len(files)} images)")
                        epoch_files.extend(files)
                        epoch_masks.extend(masks)
            else:
                # === Original logic: equal uniform sampling (backward compatible) ===
                epoch_files, epoch_masks = [], []
                for dataset_name, (files, masks) in labeled_datasets.items():
                    n_samples = min(len(files), max_samples_per_epoch) if max_samples_per_epoch is not None else len(files)
                    if n_samples > 0:
                        indices = np.random.permutation(len(files))[:n_samples]
                        epoch_files.extend([files[i] for i in indices])
                        epoch_masks.extend([masks[i] for i in indices])

            epoch_dataset = CellposeDataset(
                epoch_files, epoch_masks,
                transform=BatchTransform(bsize=bsize, scale_range=scale_range, crop_mode=crop_mode),
                normalize_params=normalize_params,
                device=device,
                add_noise_files=add_noise_files
            )
            if use_distributed:
                epoch_sampler = torch.utils.data.distributed.DistributedSampler(epoch_dataset)
                epoch_sampler.set_epoch(iepoch)
                current_train_loader = DataLoader(
                    epoch_dataset,
                    batch_size=batch_size,
                    sampler=epoch_sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=collate_skip_invalid,
                    drop_last=True
                )
            else:
                current_train_loader = DataLoader(
                    epoch_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate_skip_invalid,
                    drop_last=True
                )
            current_nimg = len(epoch_files)
        elif max_samples_per_epoch is not None and max_samples_per_epoch > 0 and max_samples_per_epoch < len(train_files):
            indices = np.random.permutation(len(train_files))[:max_samples_per_epoch]
            epoch_files = [train_files[i] for i in indices]
            epoch_masks = [train_mask_files[i] for i in indices]
            epoch_dataset = CellposeDataset(
                epoch_files, epoch_masks,
                transform=BatchTransform(bsize=bsize, scale_range=scale_range, crop_mode=crop_mode),
                normalize_params=normalize_params,
                device=device,
                add_noise_files=add_noise_files
            )
            if use_distributed:
                epoch_sampler = torch.utils.data.distributed.DistributedSampler(epoch_dataset)
                epoch_sampler.set_epoch(iepoch)
                current_train_loader = DataLoader(
                    epoch_dataset,
                    batch_size=batch_size,
                    sampler=epoch_sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=collate_skip_invalid,
                    drop_last=True
                )
            else:
                current_train_loader = DataLoader(
                    epoch_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate_skip_invalid,
                    drop_last=True
                )
            current_nimg = len(epoch_files)
        else:
            current_train_loader = train_loader
            current_nimg = nimg
            # DDP needs to set sampler per epoch
            if use_distributed:
                train_loader.sampler.set_epoch(iepoch)

        # ---- Print epoch info ----
        if not use_distributed or dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {iepoch}/{n_epochs}")
            if labeled_datasets is not None:
                print(f"Sampled: {current_nimg} labeled")
            print(f"{'='*60}")

        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]

        net.train()

        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(current_train_loader)):
            if batch is None:
                continue
            if len(batch) == 2:
                images, masks = batch

                # Convert data types
                images = images.to(device).to(model_dtype)
                masks = masks.to(device).to(model_dtype)

                # Forward pass
                outputs = net(images)[0]

                veci = 5.0 * masks[:, -2:]  # lbl[:, -2:] corresponds to flowsY, flowsX
                loss_flow = mse_loss(outputs[:, -3:-1], veci)  # y[:, -3:-1] corresponds to flowsY, flowsX
                loss_flow /= 2.0

                # Compute cellprob loss
                loss_cell = bce_loss(outputs[:, -1], (masks[:, -3] > 0.5).to(outputs.dtype))

                # Combine losses
                loss = loss_flow + loss_cell

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record loss (consistent with original code)
                train_loss = loss.item()
                train_loss *= len(images)

                lavg += train_loss
                nsum += len(images)
                train_losses[iepoch] += train_loss
                epoch_loss += train_loss
                num_batches += 1

        # DDP mode: synchronize training loss
        if use_distributed:
            train_loss_tensor = torch.tensor([train_losses[iepoch]], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_losses[iepoch] = train_loss_tensor.item() / world_size
            nsum_tensor = torch.tensor([nsum], device=device)
            dist.all_reduce(nsum_tensor, op=dist.ReduceOp.SUM)
            nsum = nsum_tensor.item()

        train_losses[iepoch] /= current_nimg

        # Test (every 10 epochs or at epoch 5)
        lavgt = 0
        if iepoch == 5 or iepoch % 10 == 0:
            test_samples = 0

            if test_loader is not None:
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

                        if outputs.shape[1] > 3:
                            loss3 = ce_loss(outputs[:, :-3], masks[:, 0].long())
                            loss += loss3

                        test_loss = loss.item()
                        test_loss *= len(images)
                        lavgt += test_loss
                        test_samples += len(images)

                lavgt /= test_samples if test_samples > 0 else 1
                test_losses[iepoch] = lavgt

        # Print results (only on rank 0)
        if not use_distributed or dist.get_rank() == 0:
            lavg_epoch = lavg / nsum if nsum > 0 else 0
            print(
                f"{iepoch}, train_loss={lavg_epoch:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time() - t0:.2f}s")
        lavg, nsum = 0, 0

        # Save model (only on rank 0)
        if not use_distributed or dist.get_rank() == 0:
            if (iepoch + 1) % save_interval == 0 or iepoch == n_epochs - 1:
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
                print(f"saving network parameters to {filename0}")

                # Use net.module to get the original model when saving
                save_net = net.module if isinstance(net, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)) else net
                if hasattr(save_net, 'save_model'):
                    save_net.save_model(filename0)
                else:
                    torch.save(save_net.state_dict(), str(filename0) + ".pt")

        # DDP mode: synchronize all processes after saving
        if use_distributed:
            dist.barrier()

    # Final model verification (only on rank 0)
    if not use_distributed or dist.get_rank() == 0:
        final_filename = str(filename) + f"_epoch_{n_epochs-1:04d}"
        if Path(final_filename).exists():
            file_size = Path(final_filename).stat().st_size
            print(f"\n[Rank 0] Final model verified: {final_filename} ({file_size / 1024 / 1024:.2f} MB)")
        elif Path(final_filename + ".pt").exists():
            file_size = Path(final_filename + ".pt").stat().st_size
            print(f"\n[Rank 0] Final model verified: {final_filename}.pt ({file_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"\n[Rank 0] WARNING: Final model NOT FOUND!")

    if use_distributed:
        dist.barrier()

    return filename, train_losses, test_losses


def semi_train_seg(net, train_files, train_mask_files=None,
                   unlabeled_files=None, unlabeled_mask_files=None,
                   test_files=None, test_mask_files=None,
                   batch_size=1, n_epochs=100,
                   learning_rate=1e-5, weight_decay=0.1, bsize=256,
                   save_path=None, models_dir=None, model_name=None, device=None,
                   num_workers=0, nimg_per_epoch=None, use_distributed=False):
    """
    Semi-supervised training function: perform supervised training (labeled data) first in each epoch,
    then unsupervised consistency training (unlabeled data).
    Data loading is consistent with train_seg_improved, using DataLoader for streaming loading.

    Args:
        net: network model
        train_files: list of labeled training image file paths
        train_mask_files: list of labeled training mask file paths
        unlabeled_files: list of unlabeled image file paths (optional)
        test_files: list of test image file paths (optional)
        test_mask_files: list of test mask file paths (optional)
        batch_size: batch size
        n_epochs: number of training epochs
        learning_rate: learning rate
        weight_decay: weight decay
        bsize: crop size
        save_path: model save path
        model_name: model name
        device: computing device
        num_workers: number of DataLoader workers
        nimg_per_epoch: number of images per epoch (None means all)

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

    # ---- Labeled dataset ----
    train_dataset = CellposeDataset(
        train_files, train_mask_files,
        transform=BatchTransform(bsize=bsize),
        normalize_params=normalize_params,
        device=device
    )

    nimg = len(train_dataset)
    nimg_per_epoch = nimg if nimg_per_epoch is None else min(nimg_per_epoch, nimg)

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

    # ---- Unlabeled dataset ----
    unlabeled_loader = None
    n_unlabeled = 0
    if unlabeled_files is not None and len(unlabeled_files) > 0:
        n_unlabeled = len(unlabeled_files)
        unlabeled_dataset = CellposeDataset(
            unlabeled_files, mask_files=unlabeled_mask_files,
            transform=BatchTransform(bsize=bsize),
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

    # ---- Test dataset ----
    test_loader = None
    test_dataset = None
    if test_files is not None:
        test_dataset = CellposeDataset(
            test_files, test_mask_files,
            transform=BatchTransform(bsize=bsize),
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

    # ---- Learning rate schedule ----
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

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

    # ---- Loss function ----
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    filename = models_dir / model_name

    # Print only on rank 0
    if not use_distributed or dist.get_rank() == 0:
        print(f">>> n_epochs={n_epochs}, n_train={nimg}, n_unlabeled={n_unlabeled}")
        if test_dataset is not None:
            print(f">>> n_test={len(test_dataset)}")
        print(f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")
        print(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses = np.zeros(n_epochs)
    semi_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    # Get model dtype for data conversion
    if use_distributed:
        actual_net_for_dtype = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
    else:
        actual_net_for_dtype = net.module if isinstance(net, torch.nn.DataParallel) else net
    model_dtype = actual_net_for_dtype.dtype

    for iepoch in range(n_epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]
        
        # DDP needs to set sampler per epoch
        if use_distributed:
            train_loader.sampler.set_epoch(iepoch)
            if unlabeled_loader is not None:
                unlabeled_loader.sampler.set_epoch(iepoch)

        net.train()

        # ============ Phase 1: Supervised training ============
        for batch in tqdm(train_loader, desc=f"Epoch {iepoch} [Sup]"):
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

        train_losses[iepoch] /= max(nimg_per_epoch, 1)

        # ============ Phase 2: Semi-supervised training (unlabeled data consistency regularization) ============
        if unlabeled_loader is not None:
            from cellpose.denoise import add_noise
            nchan_noise = 1
            diam_mean = 30
            semi_nsum = 0

            for batch in tqdm(unlabeled_loader, desc=f"Epoch {iepoch} [Semi]"):
                if batch is None:
                    continue
                images, masks = batch
                images = images.to(device).to(model_dtype)
                masks = masks.to(device).to(model_dtype)

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

                # Noisy image prediction (backward pass)
                y_noisy = net(images_noise)[0]

                veci = 5.0 * masks[:, -2:]
                loss_flow = mse_loss(y_noisy[:, -3:-1], veci) / 2.0
                loss_cell = bce_loss(y_noisy[:, -1], (masks[:, -3] > 0.5).to(y_noisy.dtype))
                loss = loss_flow + loss_cell

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                semi_loss = loss.item() * n_imgs
                semi_losses[iepoch] += semi_loss
                semi_nsum += n_imgs

            semi_losses[iepoch] /= max(semi_nsum, 1)

        # ============ Test evaluation ============
        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0
            test_samples = 0

            if test_loader is not None:
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
            print(
                f"{iepoch}, train_loss={lavg_epoch:.4f}, semi_loss={semi_losses[iepoch]:.4f}, "
                f"test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time() - t0:.2f}s"
            )
            lavg, nsum = 0, 0

        # ============ Save model ============
        if iepoch == n_epochs - 1 or (iepoch % 10 == 0 and iepoch != 0):
            if iepoch != n_epochs - 1:
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            print(f"saving network parameters to {filename0}")

            if hasattr(net, 'save_model'):
                net.save_model(filename0)
            else:
                torch.save(net.state_dict(), str(filename0) + ".pt")

    # Save final model
    if hasattr(net, 'save_model'):
        net.save_model(filename)
    else:
        torch.save(net.state_dict(), str(filename) + ".pt")

    return filename, train_losses, semi_losses, test_losses

