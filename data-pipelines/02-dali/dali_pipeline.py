"""
dali_pipeline.py - NVIDIA DALI for GPU-Accelerated Data Loading

DALI moves data preprocessing to the GPU, eliminating CPU bottlenecks.
Benefits:
- 2-10x faster image decoding
- GPU-native augmentations
- Built-in prefetching and pipelining

Requirements:
    pip install nvidia-dali-cuda120  # Match your CUDA version

Author: CUDA Lab
"""

import torch
import numpy as np
import time
import argparse
from typing import Optional, Tuple
import os

# Check DALI availability
try:
    from nvidia.dali import pipeline_def, fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    import nvidia.dali.types as types
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("NVIDIA DALI not available. Install with: pip install nvidia-dali-cuda120")


if DALI_AVAILABLE:
    
    @pipeline_def
    def image_pipeline(
        data_dir: str,
        crop_size: int = 224,
        random_shuffle: bool = True,
        device_id: int = 0
    ):
        """
        GPU-accelerated image classification pipeline.
        
        Performs on GPU:
        - JPEG decoding (nvJPEG)
        - Random crop
        - Random flip
        - Normalization
        """
        # Read and decode on GPU
        jpegs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=random_shuffle,
            name="Reader"
        )
        
        # Decode directly to GPU
        images = fn.decoders.image(
            jpegs,
            device="mixed",  # CPU read, GPU decode
            output_type=types.RGB
        )
        
        # GPU augmentations
        images = fn.random_resized_crop(
            images,
            size=crop_size,
            random_area=[0.08, 1.0],
            random_aspect_ratio=[0.75, 1.333],
            device="gpu"
        )
        
        images = fn.flip(
            images,
            horizontal=fn.random.coin_flip(probability=0.5),
            device="gpu"
        )
        
        # Normalize
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            device="gpu"
        )
        
        return images, labels
    
    
    @pipeline_def
    def synthetic_pipeline(
        batch_size: int,
        image_size: int = 224,
        num_classes: int = 1000,
        device_id: int = 0
    ):
        """
        Synthetic data pipeline for benchmarking.
        Generates random images directly on GPU.
        """
        # Generate random images on GPU
        images = fn.random.uniform(
            range=[0, 255],
            shape=[3, image_size, image_size],
            dtype=types.FLOAT,
            device="gpu"
        )
        
        # Normalize
        images = fn.normalize(
            images,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            stddev=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            device="gpu"
        )
        
        # Random labels
        labels = fn.random.uniform(
            range=[0, num_classes],
            dtype=types.INT32,
            device="cpu"
        )
        labels = fn.reshape(labels, shape=[1])
        
        return images, labels
    
    
    @pipeline_def
    def video_pipeline(
        video_files: list,
        sequence_length: int = 16,
        stride: int = 1,
        device_id: int = 0
    ):
        """
        GPU-accelerated video loading pipeline.
        Useful for video understanding tasks.
        """
        videos = fn.readers.video(
            filenames=video_files,
            sequence_length=sequence_length,
            stride=stride,
            device="gpu",
            name="VideoReader"
        )
        
        # Resize frames
        videos = fn.resize(
            videos,
            size=[224, 224],
            device="gpu"
        )
        
        # Normalize
        videos = fn.normalize(
            videos,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            stddev=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            device="gpu"
        )
        
        return videos
    
    
    class DALIDataLoader:
        """
        Wrapper to make DALI pipelines look like PyTorch DataLoaders.
        """
        
        def __init__(
            self,
            pipeline,
            output_map: list = ["images", "labels"],
            auto_reset: bool = True,
            size: int = -1
        ):
            self.pipeline = pipeline
            self.pipeline.build()
            
            self.iterator = DALIGenericIterator(
                [self.pipeline],
                output_map,
                auto_reset=auto_reset,
                last_batch_policy=LastBatchPolicy.DROP,
                size=size
            )
        
        def __iter__(self):
            return self
        
        def __next__(self):
            batch = next(self.iterator)
            # DALI returns list of dicts, extract first
            return batch[0]["images"], batch[0]["labels"]
        
        def __len__(self):
            return self.iterator._size // self.pipeline.max_batch_size
        
        def reset(self):
            self.iterator.reset()


def create_dali_dataloader(
    data_dir: str = None,
    batch_size: int = 32,
    num_threads: int = 4,
    device_id: int = 0,
    image_size: int = 224,
    synthetic: bool = False
) -> 'DALIDataLoader':
    """Create a DALI-backed DataLoader."""
    
    if not DALI_AVAILABLE:
        raise RuntimeError("DALI not available")
    
    if synthetic or data_dir is None:
        pipeline = synthetic_pipeline(
            batch_size=batch_size,
            image_size=image_size,
            device_id=device_id,
            num_threads=num_threads
        )
    else:
        pipeline = image_pipeline(
            data_dir=data_dir,
            crop_size=image_size,
            device_id=device_id,
            batch_size=batch_size,
            num_threads=num_threads
        )
    
    return DALIDataLoader(pipeline)


def benchmark_dali_vs_pytorch(
    batch_size: int = 32,
    n_batches: int = 100,
    image_size: int = 224
):
    """Compare DALI vs PyTorch DataLoader throughput."""
    
    print("\n" + "="*60)
    print("DALI vs PyTorch DataLoader Benchmark")
    print("="*60)
    
    device = torch.device('cuda')
    
    # PyTorch DataLoader with synthetic data
    print("\nPyTorch DataLoader:")
    
    from torch.utils.data import Dataset, DataLoader
    
    class SyntheticDataset(Dataset):
        def __init__(self, size, image_size):
            self.size = size
            self.image_size = image_size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Simulate image loading and preprocessing
            img = torch.randn(3, self.image_size, self.image_size)
            label = torch.randint(0, 1000, (1,)).item()
            return img, label
    
    dataset = SyntheticDataset(n_batches * batch_size, image_size)
    pytorch_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Warmup
    for i, (x, y) in enumerate(pytorch_loader):
        x = x.to(device, non_blocking=True)
        if i >= 10:
            break
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i, (x, y) in enumerate(pytorch_loader):
        x = x.to(device, non_blocking=True)
        torch.cuda.synchronize()
        if i >= n_batches:
            break
    
    pytorch_time = time.perf_counter() - start
    pytorch_throughput = (n_batches * batch_size) / pytorch_time
    
    print(f"  Throughput: {pytorch_throughput:,.0f} samples/sec")
    print(f"  Time: {pytorch_time:.2f}s for {n_batches} batches")
    
    # DALI benchmark
    if DALI_AVAILABLE:
        print("\nDALI DataLoader:")
        
        dali_loader = create_dali_dataloader(
            batch_size=batch_size,
            image_size=image_size,
            synthetic=True
        )
        
        # Warmup
        for i, (x, y) in enumerate(dali_loader):
            if i >= 10:
                break
        dali_loader.reset()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for i, (x, y) in enumerate(dali_loader):
            torch.cuda.synchronize()
            if i >= n_batches:
                break
        
        dali_time = time.perf_counter() - start
        dali_throughput = (n_batches * batch_size) / dali_time
        
        print(f"  Throughput: {dali_throughput:,.0f} samples/sec")
        print(f"  Time: {dali_time:.2f}s for {n_batches} batches")
        
        speedup = dali_throughput / pytorch_throughput
        print(f"\nDALI Speedup: {speedup:.2f}x")
    else:
        print("\nDALI not available for comparison")


def main():
    parser = argparse.ArgumentParser(description='DALI Pipeline Examples')
    parser.add_argument('--benchmark', action='store_true',
                       help='Compare DALI vs PyTorch')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--n-batches', type=int, default=100)
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA required for DALI benchmark")
        return
    
    if args.benchmark:
        benchmark_dali_vs_pytorch(
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            image_size=args.image_size
        )
    else:
        # Default: just show benchmark
        benchmark_dali_vs_pytorch()


if __name__ == "__main__":
    main()
