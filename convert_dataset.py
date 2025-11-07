#!/usr/bin/env python3
"""
Convert dataset from masks to bounding boxes for Ground-A-Video model.
This script processes video frames and masks, extracting frames and converting
masks to bounding boxes in YAML config format.
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
import shutil


# ============================================================================
# CONFIG SECTION - Customize settings for each video sample
# ============================================================================

# Global default settings (used if not specified for a specific sample)
DEFAULT_CONFIG = {
    'n_sample_frames': 8,        # Number of frames to sample
    'start_sample_frame': 0,      # Starting frame index (0-based)
    'sampling_rate': 1,           # Take 1 frame every N frames
}

# Per-sample configuration
# Format: 'category/sample_name': {config_dict}
# If a sample is not listed here, DEFAULT_CONFIG will be used
SAMPLE_CONFIGS = {
    # Example configurations:
    'o2o/3ball': {
        'n_sample_frames': 15,
        'start_sample_frame': 0,
        'sampling_rate': 4,
    },
    'o2o/car_truck': {
        'n_sample_frames': 15,
        'start_sample_frame': 0,
        'sampling_rate': 6,
    },
    'o2o/cloth-bag': {
        'n_sample_frames': 15,
        'start_sample_frame': 70,
        'sampling_rate': 5,
    },
    'o2o/dog-cat': {
        'n_sample_frames': 15,
        'start_sample_frame': 30,   
        'sampling_rate': 6,
    },
    'o2o/elephant': {
        'n_sample_frames': 15,
        'start_sample_frame': 100,
        'sampling_rate': 6,
    },
    'o2o/hair': {
        'n_sample_frames': 15,
        'start_sample_frame': 5,
        'sampling_rate': 2,
    },
    'o2p/Charlie-pillar': {
        'n_sample_frames': 15,
        'start_sample_frame': 60,
        'sampling_rate': 3,
    },
    'o2p/man-obj-horizontal': {
        'n_sample_frames': 15,
        'start_sample_frame': 40,
        'sampling_rate': 4,
    },
    'o2p/woman_car': {
        'n_sample_frames': 15,
        'start_sample_frame': 20,
        'sampling_rate': 4,
    },
    'p2p/chinese_palace': {
        'n_sample_frames': 15,
        'start_sample_frame': 0,
        'sampling_rate': 5,
    },
    'p2p/lalaland': {
        'n_sample_frames': 15,
        'start_sample_frame': 0,
        'sampling_rate': 4,
    },
    'p2p/n_u_c_s': {
        'n_sample_frames': 12,
        'start_sample_frame': 5,
        'sampling_rate': 2,
    },
    'p2p/u_name': {
        'n_sample_frames': 15,
        'start_sample_frame': 0,
        'sampling_rate': 5,
    },
    'p2p/02_walk_walk': {
        'n_sample_frames': 15,
        'start_sample_frame': 0,
        'sampling_rate': 4,
    },
    'p2p/03_withwalk': {
        'n_sample_frames': 15,
        'start_sample_frame': 0,
        'sampling_rate': 4,
    },
}

# Base assets directory path
ASSETS_DIR = 'assets'  # Can be absolute or relative path

# ============================================================================
# END OF CONFIG SECTION
# ============================================================================


def get_bbox_from_mask(mask):
    """
    Extract bounding box from binary mask.

    Args:
        mask: Binary mask (numpy array)

    Returns:
        [x1, y1, x2, y2] normalized coordinates or None if mask is empty
    """
    # Find non-zero pixels
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Get image dimensions
    h, w = mask.shape

    # Normalize coordinates to [0, 1]
    # Convert to Python float explicitly to avoid numpy types in YAML
    x1 = float(cmin) / float(w)
    y1 = float(rmin) / float(h)
    x2 = float(cmax + 1) / float(w)  # +1 to include the last pixel
    y2 = float(rmax + 1) / float(h)

    # Ensure coordinates are within [0, 1]
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)

    # Round and return as Python floats (not numpy types)
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def sample_frames(total_frames, n_sample_frames=8, start_sample_frame=0, sampling_rate=1):
    """
    Sample frames with custom start point and sampling rate.

    Args:
        total_frames: Total number of frames available
        n_sample_frames: Number of frames to sample
        start_sample_frame: Starting frame index (0-based)
        sampling_rate: Sample every Nth frame (e.g., 4 means take every 4th frame)

    Returns:
        List of frame indices

    Example:
        n_sample_frames=15, start_sample_frame=10, sampling_rate=4
        -> [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66]
    """
    indices = []
    current_frame = start_sample_frame

    for i in range(n_sample_frames):
        if current_frame >= total_frames:
            print(f"   ⚠️  Warning: Requested frame {current_frame} exceeds total frames {total_frames}")
            break
        indices.append(current_frame)
        current_frame += sampling_rate

    if len(indices) < n_sample_frames:
        print(f"   ⚠️  Warning: Only sampled {len(indices)} frames (requested {n_sample_frames})")

    return indices


def process_video_sample(sample_path, category, sample_name, output_base_dir,
                         n_sample_frames=8, start_sample_frame=0, sampling_rate=1):
    """
    Process a single video sample: extract frames and generate config.

    Args:
        sample_path: Path to the video sample directory
        category: Category name (o2o, o2p, p2p)
        sample_name: Sample name (e.g., '3ball')
        output_base_dir: Base output directory
        n_sample_frames: Number of frames to extract
        start_sample_frame: Starting frame index (0-based)
        sampling_rate: Sample every Nth frame

    Returns:
        True if successful, False otherwise
    """
    frames_dir = sample_path / 'frames'
    masks_dir = sample_path / 'masks'

    if not frames_dir.exists() or not masks_dir.exists():
        print(f"⚠️  Skipping {sample_name}: missing frames or masks directory")
        return False

    # Get all frame files
    frame_files = sorted([f for f in frames_dir.glob('*.jpg')])
    if not frame_files:
        frame_files = sorted([f for f in frames_dir.glob('*.png')])

    if len(frame_files) == 0:
        print(f"⚠️  Skipping {sample_name}: no frames found")
        return False

    total_frames = len(frame_files)
    print(f"\n📹 Processing {category}/{sample_name}: {total_frames} frames")
    print(f"   Config: n_sample_frames={n_sample_frames}, start_frame={start_sample_frame}, rate={sampling_rate}")

    # Sample frames
    sampled_indices = sample_frames(total_frames, n_sample_frames, start_sample_frame, sampling_rate)
    print(f"   Sampled frame indices: {sampled_indices}")

    # Get object directories (excluding 'bg' and visualization files)
    obj_dirs = sorted([d for d in masks_dir.iterdir()
                      if d.is_dir() and d.name not in ['bg', 'background']])

    if len(obj_dirs) == 0:
        print(f"⚠️  Skipping {sample_name}: no object masks found")
        return False

    print(f"   Found {len(obj_dirs)} objects: {[d.name for d in obj_dirs]}")

    # Create output directories
    output_images_dir = Path(output_base_dir) / 'video_images' / category / sample_name
    output_config_dir = Path(output_base_dir) / 'video_configs' / category

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_config_dir.mkdir(parents=True, exist_ok=True)

    # Copy sampled frames
    print(f"   Copying frames to {output_images_dir}")
    for i, frame_idx in enumerate(sampled_indices):
        src_frame = frame_files[frame_idx]
        dst_frame = output_images_dir / f"{i+1}.jpg"

        # Convert to jpg if necessary
        img = Image.open(src_frame)
        img = img.convert('RGB')
        img.save(dst_frame, 'JPEG')

    # Extract bounding boxes for each object in each sampled frame
    locations_per_frame = []  # [frame][object] -> bbox

    for frame_idx in sampled_indices:
        frame_bboxes = []

        for obj_dir in obj_dirs:
            # Get mask file for this frame
            mask_files = list(obj_dir.glob(f"{frame_files[frame_idx].stem}.*"))

            if not mask_files:
                # If no mask for this frame, use [0, 0, 0, 0]
                frame_bboxes.append([0.0, 0.0, 0.0, 0.0])
                continue

            # Read mask
            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                frame_bboxes.append([0.0, 0.0, 0.0, 0.0])
                continue

            # Get bounding box
            bbox = get_bbox_from_mask(mask)
            if bbox is None:
                bbox = [0.0, 0.0, 0.0, 0.0]

            frame_bboxes.append(bbox)

        locations_per_frame.append(frame_bboxes)

    # Transpose to get locations in the format needed by config
    # Config format: [[bbox_obj1, bbox_obj2, ...], ...] for each frame
    # We already have that format in locations_per_frame

    # Generate object names (you may want to customize this)
    obj_names = [obj_dir.name for obj_dir in obj_dirs]

    # Save config - generate YAML manually for proper formatting
    config_file = output_config_dir / f"{sample_name}.yaml"
    print(f"   Saving config to {config_file}")

    with open(config_file, 'w') as f:
        # Write basic fields
        f.write(f'ckpt: "gligen-inpainting-text-box/diffusion_pytorch_model.bin"\n')
        f.write(f'input_images_path: "video_images/{category}/{sample_name}"\n')
        f.write(f'prompt: "TODO: Add target prompt for {sample_name}"\n')
        f.write(f'source_prompt: "TODO: Add source prompt for {sample_name}"\n')

        # Write phrases
        f.write('phrases:\n')
        for frame_idx in range(n_sample_frames):
            phrase_list = str(obj_names).replace("'", "'")
            f.write(f'  - {phrase_list}\n')

        # Write locations with proper formatting
        f.write('locations:\n')
        for frame_bboxes in locations_per_frame:
            f.write('  - [')
            bbox_strs = []
            for bbox in frame_bboxes:
                # Format each bbox as [x1, y1, x2, y2]
                bbox_str = '[' + ', '.join([f'{v:.2f}' if v != 0 and v != 1 else ('0.0' if v == 0 else '1.0') for v in bbox]) + ']'
                bbox_strs.append(bbox_str)
            f.write(', '.join(bbox_strs))
            f.write(']\n')

        # Write source_phrases
        f.write('source_phrases:\n')
        for frame_idx in range(n_sample_frames):
            phrase_list = str(obj_names).replace("'", "'")
            f.write(f'  - {phrase_list}\n')

    print(f"✅ Successfully processed {category}/{sample_name}")
    return True


def get_sample_config(category, sample_name):
    """
    Get configuration for a specific sample.

    Args:
        category: Category name (e.g., 'o2o')
        sample_name: Sample name (e.g., '3ball')

    Returns:
        Dictionary with configuration parameters
    """
    sample_key = f"{category}/{sample_name}"

    # Check if there's a specific config for this sample
    if sample_key in SAMPLE_CONFIGS:
        config = DEFAULT_CONFIG.copy()
        config.update(SAMPLE_CONFIGS[sample_key])
        return config

    # Otherwise use default config
    return DEFAULT_CONFIG.copy()


def main():
    """Main processing function."""
    # Set up paths
    base_dir = Path(__file__).parent

    # Use configured assets directory (can be absolute or relative)
    if Path(ASSETS_DIR).is_absolute():
        assets_dir = Path(ASSETS_DIR)
    else:
        assets_dir = base_dir / ASSETS_DIR

    output_base_dir = base_dir

    if not assets_dir.exists():
        print(f"❌ Error: Assets directory not found at {assets_dir}")
        return

    # Get all categories
    categories = ['o2o', 'o2p', 'p2p']

    total_processed = 0
    total_failed = 0

    print("=" * 80)
    print("🎬 Starting dataset conversion for Ground-A-Video")
    print("=" * 80)
    print(f"📁 Assets directory: {assets_dir}")
    print(f"📋 Default config: n_frames={DEFAULT_CONFIG['n_sample_frames']}, "
          f"start={DEFAULT_CONFIG['start_sample_frame']}, rate={DEFAULT_CONFIG['sampling_rate']}")
    print("=" * 80)

    for category in categories:
        category_dir = assets_dir / category

        if not category_dir.exists():
            print(f"\n⚠️  Category directory not found: {category_dir}")
            continue

        print(f"\n\n{'='*80}")
        print(f"📂 Processing category: {category}")
        print(f"{'='*80}")

        # Get all video samples in this category
        samples = sorted([d for d in category_dir.iterdir() if d.is_dir()])

        print(f"Found {len(samples)} samples in {category}")

        for sample_dir in samples:
            sample_name = sample_dir.name

            # Get configuration for this sample
            config = get_sample_config(category, sample_name)

            success = process_video_sample(
                sample_dir,
                category,
                sample_name,
                output_base_dir,
                n_sample_frames=config['n_sample_frames'],
                start_sample_frame=config['start_sample_frame'],
                sampling_rate=config['sampling_rate']
            )

            if success:
                total_processed += 1
            else:
                total_failed += 1

    # Print summary
    print("\n" + "=" * 80)
    print("📊 CONVERSION SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully processed: {total_processed} samples")
    print(f"❌ Failed: {total_failed} samples")
    print(f"📁 Output directories:")
    print(f"   - Frames: {output_base_dir / 'video_images'}")
    print(f"   - Configs: {output_base_dir / 'video_configs'}")
    print("\n⚠️  NOTE: Please update the 'prompt' and 'source_prompt' fields in the")
    print("   generated YAML files with appropriate descriptions for each video.")
    print("=" * 80)


if __name__ == '__main__':
    main()
