# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
import time
import psutil
from typing import Any, Dict, List
from contextlib import contextmanager
from datetime import datetime

# ONNX Runtime imports (optional)
try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("ONNX Runtime not available. Install with: pip install onnxruntime or onnxruntime-gpu")

# TensorRT imports (optional)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. Install TensorRT for accelerated inference.")

# HDF5 imports (optional)
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("HDF5 not available. Install with: pip install h5py")

# MLflow imports (optional)
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: pip install mlflow")

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default="vit_t",
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b', 'vit_t']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="weights/mobile_sam.pt",
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--onnx-model",
    type=str,
    default=None,
    help="Path to ONNX model file. If provided, will use ONNX Runtime instead of PyTorch model.",
)

parser.add_argument(
    "--tensorrt-model", 
    type=str,
    default=None,
    help="Path to TensorRT engine file. If provided, will use TensorRT instead of PyTorch/ONNX.",
)

parser.add_argument(
    "--image-encoder-onnx",
    type=str, 
    default=None,
    help="Path to separate ONNX model for image encoder (if using split models).",
)

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

parser.add_argument(
    "--hdf5-output",
    action="store_true", 
    help=(
        "Save everything in a single HDF5 file (.h5) containing masks, metadata, "
        "original image, and visualization. This allows post-processing without "
        "rerunning segmentation. Requires h5py."
    ),
)

# Visualization options
visualization_settings = parser.add_argument_group("Visualization Settings")

visualization_settings.add_argument(
    "--visualize",
    action="store_true",
    help="Generate visualization images with mask overlays.",
)

visualization_settings.add_argument(
    "--random-colors",
    action="store_true",
    help="Use random colors for mask visualization instead of default blue.",
)

visualization_settings.add_argument(
    "--no-contours",
    action="store_true",
    help="Disable contour lines in visualizations (contours are enabled by default when --visualize is used).",
)

# MLflow tracking options
tracking_settings = parser.add_argument_group("Tracking Settings")

tracking_settings.add_argument(
    "--track-performance",
    action="store_true",
    help="Enable detailed performance tracking and logging.",
)

tracking_settings.add_argument(
    "--mlflow-tracking",
    action="store_true",
    help="Enable MLflow tracking for experiments (requires mlflow).",
)

tracking_settings.add_argument(
    "--mlflow-experiment",
    type=str,
    default="mobile_sam_inference",
    help="MLflow experiment name for tracking runs.",
)

tracking_settings.add_argument(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="MLflow run name. If not provided, will be auto-generated.",
)



@contextmanager
def timer(name: str, device: str = "cpu"):
    """Context manager for timing operations with GPU synchronization."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    start_memory = psutil.virtual_memory().used / 1024**3  # GB
    
    if device == "cuda" and torch.cuda.is_available():
        start_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    else:
        start_gpu_memory = 0
    
    try:
        yield
    finally:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        end_memory = psutil.virtual_memory().used / 1024**3  # GB
        
        if device == "cuda" and torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            end_gpu_memory = 0
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        gpu_memory_delta = end_gpu_memory - start_gpu_memory
        
        print(f"{name}: {duration:.4f}s")
        print(f"  CPU Memory: {memory_delta:+.3f}GB (total: {end_memory:.3f}GB)")
        if device == "cuda" and torch.cuda.is_available():
            print(f"  GPU Memory: {gpu_memory_delta:+.3f}GB (total: {end_gpu_memory:.3f}GB)")


class PerformanceTracker:
    """Track performance metrics for inference operations."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.metrics = {}
        self.timings = {}
        
    def start_timer(self, name: str):
        """Start timing an operation."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings[name] = {
            'start_time': time.perf_counter(),
            'start_cpu_memory': psutil.virtual_memory().used / 1024**3,
            'start_gpu_memory': torch.cuda.memory_allocated() / 1024**3 if self.device == "cuda" and torch.cuda.is_available() else 0
        }
    
    def end_timer(self, name: str):
        """End timing an operation and record metrics."""
        if name not in self.timings:
            raise ValueError(f"Timer '{name}' was not started")
            
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        timing_info = self.timings[name]
        end_time = time.perf_counter()
        end_cpu_memory = psutil.virtual_memory().used / 1024**3
        end_gpu_memory = torch.cuda.memory_allocated() / 1024**3 if self.device == "cuda" and torch.cuda.is_available() else 0
        
        duration = end_time - timing_info['start_time']
        cpu_memory_delta = end_cpu_memory - timing_info['start_cpu_memory']
        gpu_memory_delta = end_gpu_memory - timing_info['start_gpu_memory']
        
        self.metrics[name] = {
            'duration_seconds': duration,
            'cpu_memory_delta_gb': cpu_memory_delta,
            'gpu_memory_delta_gb': gpu_memory_delta,
            'final_cpu_memory_gb': end_cpu_memory,
            'final_gpu_memory_gb': end_gpu_memory
        }
        
        del self.timings[name]
        return duration
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics.copy()
    
    def log_metrics(self, prefix: str = ""):
        """Print all collected metrics."""
        for name, metrics in self.metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            print(f"{metric_name}:")
            print(f"  Duration: {metrics['duration_seconds']:.4f}s")
            print(f"  CPU Memory Delta: {metrics['cpu_memory_delta_gb']:+.3f}GB")
            if metrics['gpu_memory_delta_gb'] > 0:
                print(f"  GPU Memory Delta: {metrics['gpu_memory_delta_gb']:+.3f}GB")


amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def fast_show_mask(
    annotation,
    random_color=False,
    target_height=960,
    target_width=960,
):
    """
    CPU-based mask visualization function.
    """
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    # Sort annotation by area
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::1]
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    if random_color == True:
        color = np.random.random((mask_sum, 1, 1, 3))
    else:
        color = np.ones((mask_sum, 1, 1, 3)) * np.array(
            [30 / 255, 144 / 255, 255 / 255]
        )
    transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    mask = np.zeros((height, weight, 4))

    h_indices, w_indices = np.meshgrid(
        np.arange(height), np.arange(weight), indexing="ij"
    )
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    mask[h_indices, w_indices, :] = mask_image[indices]

    if height != target_height or weight != target_width:
        mask = cv2.resize(
            mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )

    return mask


def fast_show_mask_gpu(
    annotation,
    random_color=False,
    target_height=960,
    target_width=960,
):
    """
    GPU-based mask visualization function.
    """
    device = annotation.device
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    if random_color == True:
        color = torch.rand((mask_sum, 1, 1, 3)).to(device)
    else:
        color = torch.ones((mask_sum, 1, 1, 3)).to(device) * torch.tensor(
            [30 / 255, 144 / 255, 255 / 255]
        ).to(device)
    transparency = torch.ones((mask_sum, 1, 1, 1)).to(device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    
    mask = torch.zeros((height, weight, 4)).to(device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    
    mask[h_indices, w_indices, :] = mask_image[indices]
    mask_cpu = mask.cpu().numpy()
    
    if height != target_height or weight != target_width:
        mask_cpu = cv2.resize(
            mask_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
    return mask_cpu


def visualize_masks(
    image,
    masks,
    device,
    random_color=True,
    show_contours=True,
):
    """
    Create visualization of masks overlaid on the original image with optional contour lines.
    """
    if isinstance(masks[0], dict):
        annotations = [mask["segmentation"] for mask in masks]
    else:
        annotations = masks

    original_h, original_w = image.shape[:2]
    
    if device == "cpu":
        annotations = np.array(annotations)
        inner_mask = fast_show_mask(
            annotations,
            random_color=random_color,
            target_height=original_h,
            target_width=original_w,
        )
    else:
        if isinstance(annotations[0], np.ndarray):
            annotations = np.array(annotations)
            annotations = torch.from_numpy(annotations)
        inner_mask = fast_show_mask_gpu(
            annotations,
            random_color=random_color,
            target_height=original_h,
            target_width=original_w,
        )
    
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()

    # Convert image to PIL Image
    pil_image = Image.fromarray(image).convert("RGBA")
    
    # Create overlay
    overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), "RGBA")
    pil_image.paste(overlay_inner, (0, 0), overlay_inner)

    # Add contour lines if requested
    if show_contours:
        pil_image = add_contour_lines(pil_image, annotations, original_h, original_w)

    return pil_image


def add_contour_lines(pil_image, annotations, height, width):
    """
    Add contour lines around mask boundaries.
    """
    # Convert PIL image to numpy for cv2 operations
    img_array = np.array(pil_image.convert("RGB"))
    
    # Process each mask to draw contours
    for mask in annotations:
        # Ensure mask is the right size
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Convert to uint8 if needed
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours with white lines for visibility
        cv2.drawContours(img_array, contours, -1, (255, 255, 255), thickness=2)
        
        # Add a thin black outline for better contrast
        cv2.drawContours(img_array, contours, -1, (0, 0, 0), thickness=1)
    
    # Convert back to PIL Image
    return Image.fromarray(img_array).convert("RGBA")


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def write_hdf5_output(image, masks, viz_image, args, image_path, performance_metrics=None):
    """
    Write all segmentation data to a single HDF5 file.
    
    Structure:
    /metadata/
        - creation_time
        - image_path
        - model_type
        - amg_parameters
        - performance_metrics (optional)
    /image/
        - original (RGB image array)
        - height, width, channels as attributes
    /masks/
        - segmentations (3D array: [mask_id, height, width])
        - mask_metadata (structured array with all mask properties)
    /visualization/
        - image (RGBA visualization image)
    """
    if not HDF5_AVAILABLE:
        print("Warning: HDF5 output requested but h5py is not installed.")
        return False
    
    base = os.path.basename(image_path)
    base = os.path.splitext(base)[0]
    output_file = os.path.join(args.output, f"{base}.h5")
    
    try:
        with h5py.File(output_file, 'w') as f:
            # === METADATA GROUP ===
            meta_group = f.create_group('metadata')
            meta_group.attrs['creation_time'] = datetime.now().isoformat()
            meta_group.attrs['image_path'] = image_path
            meta_group.attrs['model_type'] = args.model_type
            meta_group.attrs['checkpoint'] = args.checkpoint
            meta_group.attrs['device'] = args.device
            
            # AMG parameters
            amg_kwargs = get_amg_kwargs(args)
            for key, value in amg_kwargs.items():
                if value is not None:
                    meta_group.attrs[f'amg_{key}'] = value
            
            # Performance metrics if available
            if performance_metrics:
                perf_group = meta_group.create_group('performance')
                for metric_name, metric_data in performance_metrics.items():
                    if isinstance(metric_data, dict):
                        metric_subgroup = perf_group.create_group(metric_name)
                        for sub_key, value in metric_data.items():
                            metric_subgroup.attrs[sub_key] = value
                    else:
                        perf_group.attrs[metric_name] = metric_data
            
            # === IMAGE GROUP ===
            img_group = f.create_group('image')
            img_dataset = img_group.create_dataset(
                'original', 
                data=image,
                compression='gzip',
                compression_opts=6
            )
            img_dataset.attrs['height'] = image.shape[0]
            img_dataset.attrs['width'] = image.shape[1] 
            img_dataset.attrs['channels'] = image.shape[2]
            img_dataset.attrs['dtype'] = str(image.dtype)
            
            # === MASKS GROUP ===
            masks_group = f.create_group('masks')
            
            # Extract segmentations and create 3D array
            if isinstance(masks[0], dict):
                segmentations = np.array([mask["segmentation"] for mask in masks])
                
                # Create structured array for metadata
                mask_fields = [
                    ('id', 'i4'),
                    ('area', 'f8'),
                    ('bbox_x', 'f8'),
                    ('bbox_y', 'f8'), 
                    ('bbox_w', 'f8'),
                    ('bbox_h', 'f8'),
                    ('point_x', 'f8'),
                    ('point_y', 'f8'),
                    ('predicted_iou', 'f8'),
                    ('stability_score', 'f8'),
                    ('crop_box_x', 'f8'),
                    ('crop_box_y', 'f8'),
                    ('crop_box_w', 'f8'),
                    ('crop_box_h', 'f8')
                ]
                
                metadata_array = np.zeros(len(masks), dtype=mask_fields)
                for i, mask_data in enumerate(masks):
                    metadata_array[i] = (
                        i,
                        mask_data["area"],
                        mask_data["bbox"][0],
                        mask_data["bbox"][1],
                        mask_data["bbox"][2], 
                        mask_data["bbox"][3],
                        mask_data["point_coords"][0][0],
                        mask_data["point_coords"][0][1],
                        mask_data["predicted_iou"],
                        mask_data["stability_score"],
                        mask_data["crop_box"][0],
                        mask_data["crop_box"][1],
                        mask_data["crop_box"][2],
                        mask_data["crop_box"][3]
                    )
            else:
                segmentations = np.array(masks)
                # Create minimal metadata for non-dict masks
                metadata_array = np.zeros(len(masks), dtype=[('id', 'i4')])
                for i in range(len(masks)):
                    metadata_array[i] = (i,)
            
            # Save segmentation masks
            seg_dataset = masks_group.create_dataset(
                'segmentations',
                data=segmentations.astype(np.uint8),
                compression='gzip',
                compression_opts=6
            )
            seg_dataset.attrs['num_masks'] = len(masks)
            seg_dataset.attrs['mask_height'] = segmentations.shape[1]
            seg_dataset.attrs['mask_width'] = segmentations.shape[2]
            
            # Save metadata
            masks_group.create_dataset(
                'metadata',
                data=metadata_array,
                compression='gzip',
                compression_opts=6
            )
            
            # === VISUALIZATION GROUP ===
            if viz_image is not None:
                viz_group = f.create_group('visualization')
                viz_array = np.array(viz_image)
                viz_dataset = viz_group.create_dataset(
                    'image',
                    data=viz_array,
                    compression='gzip',
                    compression_opts=6
                )
                viz_dataset.attrs['height'] = viz_array.shape[0]
                viz_dataset.attrs['width'] = viz_array.shape[1]
                viz_dataset.attrs['channels'] = viz_array.shape[2]
                viz_dataset.attrs['format'] = 'RGBA'
        
        print(f"  HDF5 output saved to '{output_file}'")
        return True
        
    except Exception as e:
        print(f"  Failed to save HDF5 output: {e}")
        return False


def setup_mlflow(args):
    """Setup MLflow tracking if enabled."""
    if not args.mlflow_tracking:
        return None
        
    if not MLFLOW_AVAILABLE:
        print("Warning: MLflow tracking requested but MLflow is not installed.")
        return None
    
    # Set experiment
    mlflow.set_experiment(args.mlflow_experiment)
    
    # Start run
    run_name = args.mlflow_run_name
    if run_name is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"amg_run_{timestamp}"
    
    mlflow.start_run(run_name=run_name)
    
    # Log parameters
    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("checkpoint", args.checkpoint)
    mlflow.log_param("device", args.device)
    mlflow.log_param("convert_to_rle", args.convert_to_rle)
    mlflow.log_param("visualize", args.visualize)
    mlflow.log_param("random_colors", args.random_colors)
    
    # Log AMG parameters
    amg_kwargs = get_amg_kwargs(args)
    for key, value in amg_kwargs.items():
        mlflow.log_param(f"amg_{key}", value)
    
    return True


def log_to_mlflow(metrics: Dict[str, Any], image_info: Dict[str, Any] = None):
    """Log metrics to MLflow if tracking is enabled."""
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    
    # Log timing metrics
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict):
            for sub_key, value in metric_data.items():
                mlflow.log_metric(f"{metric_name}_{sub_key}", value)
        else:
            mlflow.log_metric(metric_name, metric_data)
    
    # Log image information if provided
    if image_info:
        for key, value in image_info.items():
            mlflow.log_metric(f"image_{key}", value)


class ONNXPredictor:
    """
    ONNX-based predictor that mimics the SamPredictor interface.
    """
    def __init__(self, onnx_model_path: str, image_encoder_path: str = None, device: str = "cpu"):
        self.device = device
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None
        
        # Setup ONNX Runtime session
        providers = self._get_providers()
        try:
            self.ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
            # Check which provider was actually used
            actual_providers = self.ort_session.get_providers()
            if self.device == "cuda" and "CUDAExecutionProvider" not in actual_providers:
                print(f"Note: Falling back to CPU execution. Active providers: {actual_providers}")
            else:
                print(f"ONNX session using providers: {actual_providers}")
        except Exception as e:
            print(f"Failed to create ONNX session with providers {providers}: {e}")
            # Fallback to CPU only
            self.ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        
        # Setup image encoder if provided separately
        self.image_encoder_session = None
        if image_encoder_path:
            self.image_encoder_session = onnxruntime.InferenceSession(image_encoder_path, providers=providers)
        
        # Set up transform (assume 1024 for mobile sam)
        from mobile_sam.utils.transforms import ResizeLongestSide
        self.transform = ResizeLongestSide(1024)
        
        # Mock model attributes needed by AMG
        self.model = type('MockModel', (), {
            'mask_threshold': 0.0,
            'image_encoder': type('MockEncoder', (), {'img_size': 1024})()
        })()
    
    def _get_providers(self):
        """Get appropriate ONNX Runtime providers based on device."""
        available_providers = onnxruntime.get_available_providers()
        providers = []
        
        if self.device == "cuda":
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
                print("Using ONNX Runtime with CUDA acceleration")
            else:
                print("Warning: CUDA requested but CUDAExecutionProvider not available.")
                print(f"Available providers: {available_providers}")
                print("Install onnxruntime-gpu for CUDA support: pip install onnxruntime-gpu")
        
        providers.append("CPUExecutionProvider")
        return providers
    
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        """Set image and compute embeddings."""
        if image_format == "BGR":
            image = image[..., ::-1]
        
        # Transform image
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()[None, :, :, :].float()
        
        # Ensure exact 1024x1024 for ONNX model
        h, w = input_image_torch.shape[2], input_image_torch.shape[3]
        if h != 1024 or w != 1024:
            input_image_torch = torch.nn.functional.interpolate(
                input_image_torch, size=(1024, 1024), mode='bilinear', align_corners=False
            )
        
        # Normalize the image (SAM preprocessing)
        pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
        pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)
        input_image_norm = (input_image_torch.numpy().astype(np.float32) - pixel_mean) / pixel_std
        
        self.original_size = image.shape[:2]
        self.input_size = tuple(input_image_torch.shape[-2:])
        
        # Compute image embeddings
        if self.image_encoder_session:
            # Use separate image encoder ONNX model
            embeddings = self.image_encoder_session.run(None, {"input": input_image_norm})[0]
            self.features = embeddings
            print(f"Generated embeddings with shape: {embeddings.shape}")
        else:
            # For exported SAM models, we need to simulate image encoder output
            # The ONNX model expects embeddings, so we need to create dummy embeddings
            # In a real scenario, you'd have separate image encoder ONNX model
            # For mobile_sam with vit_t, the embedding dimension is typically 256x64x64
            embed_dim = 256
            embed_size = 64
            self.features = np.random.randn(1, embed_dim, embed_size, embed_size).astype(np.float32)
            print("Warning: Using dummy embeddings. For proper ONNX inference, export image encoder separately.")
            print("Use scripts/export_mobile_sam_onnx.py to create separate encoder and decoder ONNX files.")
        
        self.is_image_set = True
    
    def predict_torch(self, point_coords, point_labels, multimask_output=True, return_logits=False):
        """Predict masks using ONNX model."""
        if not self.is_image_set:
            raise RuntimeError("Image must be set before prediction")
        
        # Prepare inputs for ONNX model
        point_coords_np = point_coords.cpu().numpy()
        point_labels_np = point_labels.cpu().numpy()
        
        # ONNX model expects batch size 1, so we need to process one point at a time
        batch_size, num_points, _ = point_coords_np.shape
        
        all_masks = []
        all_iou_preds = []
        
        for i in range(batch_size):
            # Process each point individually
            single_point_coords = point_coords_np[i:i+1]  # Keep batch dimension
            single_point_labels = point_labels_np[i:i+1]  # Keep batch dimension
            
            # Prepare ONNX inputs (these match the export script format)
            onnx_inputs = {
                "image_embeddings": self.features,
                "point_coords": single_point_coords.astype(np.float32),
                "point_labels": single_point_labels.astype(np.float32),
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),  # No mask input
                "has_mask_input": np.array([0.0], dtype=np.float32),
                "orig_im_size": np.array(self.original_size, dtype=np.float32),
            }
            
            # Run inference
            outputs = self.ort_session.run(None, onnx_inputs)
            masks, iou_predictions = outputs[0], outputs[1]
            
            all_masks.append(masks)
            all_iou_preds.append(iou_predictions)
        
        # Stack results
        stacked_masks = np.concatenate(all_masks, axis=0)
        stacked_iou = np.concatenate(all_iou_preds, axis=0)
        
        # Convert back to torch tensors
        masks_torch = torch.from_numpy(stacked_masks)
        iou_torch = torch.from_numpy(stacked_iou)
        
        # Return format matching SamPredictor
        return masks_torch, iou_torch, None


class TensorRTPredictor:
    """
    TensorRT-based predictor that mimics the SamPredictor interface.
    """
    def __init__(self, engine_path: str, image_encoder_path: str = None, device: str = "cuda"):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        
        self.device = device
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Setup image encoder if provided (can be ONNX for now)
        self.image_encoder_session = None
        if image_encoder_path and ONNXRUNTIME_AVAILABLE:
            providers = ["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"]
            self.image_encoder_session = onnxruntime.InferenceSession(image_encoder_path, providers=providers)
        
        # Set up transform
        from mobile_sam.utils.transforms import ResizeLongestSide
        self.transform = ResizeLongestSide(1024)
        
        # Mock model attributes
        self.model = type('MockModel', (), {
            'mask_threshold': 0.0,
            'image_encoder': type('MockEncoder', (), {'img_size': 1024})()
        })()
    
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        """Set image and compute embeddings."""
        if image_format == "BGR":
            image = image[..., ::-1]
        
        # Transform and normalize image
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()[None, :, :, :].float()
        
        # Ensure exact 1024x1024 for ONNX model
        h, w = input_image_torch.shape[2], input_image_torch.shape[3]
        if h != 1024 or w != 1024:
            input_image_torch = torch.nn.functional.interpolate(
                input_image_torch, size=(1024, 1024), mode='bilinear', align_corners=False
            )
        
        pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
        pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)
        input_image_norm = (input_image_torch.numpy().astype(np.float32) - pixel_mean) / pixel_std
        
        self.original_size = image.shape[:2]
        self.input_size = tuple(input_image_torch.shape[-2:])
        
        # Compute embeddings using ONNX image encoder for now
        if self.image_encoder_session:
            embeddings = self.image_encoder_session.run(None, {"input": input_image_norm})[0]
            self.features = embeddings
            print(f"Generated embeddings with shape: {embeddings.shape}")
        else:
            # Dummy embeddings
            embed_dim, embed_size = 256, 64
            self.features = np.random.randn(1, embed_dim, embed_size, embed_size).astype(np.float32)
            print("Warning: Using dummy embeddings for TensorRT. Provide image encoder ONNX file.")
        
        self.is_image_set = True
    
    def predict_torch(self, point_coords, point_labels, multimask_output=True, return_logits=False):
        """Predict masks using TensorRT engine."""
        if not self.is_image_set:
            raise RuntimeError("Image must be set before prediction")
        
        # For now, fall back to ONNX Runtime for TensorRT
        # Full TensorRT implementation would require more complex memory management
        print("Warning: TensorRT predict_torch not fully implemented, using dummy outputs")
        
        # Return dummy outputs with correct shape
        batch_size = point_coords.shape[0]
        num_masks = 3 if multimask_output else 1
        h, w = self.original_size
        
        masks = torch.zeros(batch_size, num_masks, h, w)
        iou_preds = torch.ones(batch_size, num_masks) * 0.5
        
        return masks, iou_preds, None


class ONNXAutomaticMaskGenerator:
    """
    ONNX-based automatic mask generator that mimics SamAutomaticMaskGenerator.
    """
    def __init__(self, onnx_predictor: ONNXPredictor, **kwargs):
        self.predictor = onnx_predictor
        
        # Set default parameters (matching SamAutomaticMaskGenerator)
        self.points_per_side = kwargs.get("points_per_side", 32)
        self.points_per_batch = kwargs.get("points_per_batch", 64) 
        self.pred_iou_thresh = kwargs.get("pred_iou_thresh", 0.88)
        self.stability_score_thresh = kwargs.get("stability_score_thresh", 0.95)
        self.stability_score_offset = kwargs.get("stability_score_offset", 1.0)
        self.box_nms_thresh = kwargs.get("box_nms_thresh", 0.7)
        self.crop_n_layers = kwargs.get("crop_n_layers", 0)
        self.crop_nms_thresh = kwargs.get("crop_nms_thresh", 0.7)
        self.crop_overlap_ratio = kwargs.get("crop_overlap_ratio", 512 / 1500)
        self.crop_n_points_downscale_factor = kwargs.get("crop_n_points_downscale_factor", 1)
        self.min_mask_region_area = kwargs.get("min_mask_region_area", 0)
        self.output_mode = kwargs.get("output_mode", "binary_mask")
    
    def generate(self, image: np.ndarray):
        """Generate masks for the entire image using ONNX model."""
        # This is a simplified implementation
        # Full implementation would need all the crop handling, NMS, etc. from the original
        
        self.predictor.set_image(image)
        h, w = image.shape[:2]
        
        # Generate point grid
        if self.points_per_side:
            points_scale = np.array([[w, h]])
            points_for_image = self._build_point_grid(self.points_per_side)
            points_for_image = points_for_image * points_scale
        
        # Process in batches
        masks_list = []
        for i in range(0, len(points_for_image), self.points_per_batch):
            batch_points = points_for_image[i:i + self.points_per_batch]
            batch_masks = self._process_batch(batch_points, (h, w))
            masks_list.extend(batch_masks)
        
        return masks_list
    
    def _build_point_grid(self, n_per_side: int) -> np.ndarray:
        """Build a grid of points for mask generation."""
        offset = 1 / (2 * n_per_side)
        points_one_side = np.linspace(offset, 1 - offset, n_per_side)
        points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
        points_y = np.tile(points_one_side[:, None], (1, n_per_side))
        points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
        return points
    
    def _process_batch(self, points: np.ndarray, im_size):
        """Process a batch of points."""
        # Convert to torch tensors
        points_torch = torch.as_tensor(points, device=self.predictor.device).unsqueeze(1)
        labels_torch = torch.ones(len(points), device=self.predictor.device).unsqueeze(1)
        
        # Get predictions
        masks, iou_preds, _ = self.predictor.predict_torch(
            points_torch, labels_torch, multimask_output=True, return_logits=True
        )
        
        # Convert to format expected by AMG
        masks_list = []
        for i in range(len(points)):
            for j in range(masks.shape[1]):  # Multiple masks per point
                mask = masks[i, j].cpu().numpy() > 0.5
                masks_list.append({
                    "segmentation": mask,
                    "area": int(mask.sum()),
                    "bbox": self._mask_to_bbox(mask),
                    "point_coords": [points[i].tolist()],
                    "predicted_iou": float(iou_preds[i, j].cpu()),
                    "stability_score": 1.0,  # Simplified
                    "crop_box": [0, 0, im_size[1], im_size[0]],
                })
        
        return masks_list
    
    def _mask_to_bbox(self, mask):
        """Convert mask to bounding box."""
        if not mask.any():
            return [0, 0, 0, 0]
        rows, cols = np.where(mask)
        return [int(cols.min()), int(rows.min()), int(cols.max() - cols.min()), int(rows.max() - rows.min())]


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    # Setup MLflow tracking
    mlflow_enabled = setup_mlflow(args)
    
    # Initialize performance tracker
    tracker = PerformanceTracker(device=args.device) if args.track_performance else None
    
    try:
        # Model loading with timing
        print("Loading model...")
        if tracker:
            tracker.start_timer("model_loading")
        
        # Determine which model type to load
        if args.onnx_model or args.tensorrt_model:
            # ONNX/TensorRT model path
            if args.tensorrt_model and TENSORRT_AVAILABLE:
                print(f"Using TensorRT model: {args.tensorrt_model}")
                # TensorRT support - basic implementation
                predictor = TensorRTPredictor(
                    engine_path=args.tensorrt_model,
                    image_encoder_path=args.image_encoder_onnx,
                    device=args.device
                )
                output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
                amg_kwargs = get_amg_kwargs(args)
                amg_kwargs["output_mode"] = output_mode
                generator = ONNXAutomaticMaskGenerator(predictor, **amg_kwargs)
            elif args.onnx_model and ONNXRUNTIME_AVAILABLE:
                print(f"Using ONNX model: {args.onnx_model}")
                predictor = ONNXPredictor(
                    onnx_model_path=args.onnx_model,
                    image_encoder_path=args.image_encoder_onnx,
                    device=args.device
                )
                output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
                amg_kwargs = get_amg_kwargs(args)
                amg_kwargs["output_mode"] = output_mode
                generator = ONNXAutomaticMaskGenerator(predictor, **amg_kwargs)
            else:
                available_runtimes = []
                if ONNXRUNTIME_AVAILABLE:
                    available_runtimes.append("ONNX Runtime")
                if TENSORRT_AVAILABLE:
                    available_runtimes.append("TensorRT")
                
                if not available_runtimes:
                    raise RuntimeError("Neither ONNX Runtime nor TensorRT is available. Please install one of them.")
                else:
                    raise RuntimeError(f"Model file not found or unsupported. Available runtimes: {', '.join(available_runtimes)}")
        else:
            # PyTorch model (original behavior)
            print(f"Using PyTorch model: {args.model_type}")
            sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
            _ = sam.to(device=args.device)
            output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
            amg_kwargs = get_amg_kwargs(args)
            generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
        
        if tracker:
            tracker.end_timer("model_loading")
            print(f"Model loaded in {tracker.metrics['model_loading']['duration_seconds']:.4f}s")

        if not os.path.isdir(args.input):
            targets = [args.input]
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = [os.path.join(args.input, f) for f in targets]

        os.makedirs(args.output, exist_ok=True)
        
        # Track total processing time
        if tracker:
            tracker.start_timer("total_processing")

        total_inference_time = 0
        total_images = 0
        total_masks = 0

        for t in targets:
            print(f"Processing '{t}'...")
            
            # Image loading with timing
            if tracker:
                tracker.start_timer("image_loading")
            
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if tracker:
                tracker.end_timer("image_loading")
            
            # Get image information
            image_height, image_width = image.shape[:2]
            image_info = {
                "height": image_height,
                "width": image_width,
                "total_pixels": image_height * image_width
            }

            # Main inference with detailed timing
            print(f"  Running inference on {image_width}x{image_height} image...")
            
            if tracker:
                tracker.start_timer("inference")
                masks = generator.generate(image)
                inference_time = tracker.end_timer("inference")
                total_inference_time += inference_time
                print(f"  Inference completed in {inference_time:.4f}s")
            elif args.track_performance:
                # Simple timing even without full tracking
                with timer("Inference", args.device):
                    masks = generator.generate(image)
            else:
                masks = generator.generate(image)
            
            total_images += 1
            num_masks = len(masks)
            total_masks += num_masks
            print(f"  Generated {num_masks} masks")

            base = os.path.basename(t)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(args.output, base)
            
            # Generate visualization for HDF5 output or if explicitly requested
            viz_image = None
            if args.hdf5_output or args.visualize:
                print(f"  Generating visualization...")
                if tracker:
                    tracker.start_timer("visualization")
                
                try:
                    viz_image = visualize_masks(
                        image,
                        masks,
                        args.device,
                        random_color=args.random_colors,
                        show_contours=not args.no_contours,
                    )
                    
                    # Save standalone visualization if explicitly requested
                    if args.visualize and not args.hdf5_output:
                        viz_filename = f"{base}_visualization.png"
                        viz_path = os.path.join(args.output, viz_filename)
                        viz_image.save(viz_path)
                        print(f"  Visualization saved to '{viz_path}'")
                    
                except Exception as e:
                    print(f"  Failed to generate visualization: {e}")
                    viz_image = None
                
                if tracker:
                    tracker.end_timer("visualization")
            
            # Save output with timing
            if tracker:
                tracker.start_timer("output_saving")
            
            # HDF5 output (comprehensive single file)
            if args.hdf5_output:
                performance_data = tracker.get_metrics() if tracker else None
                write_hdf5_output(image, masks, viz_image, args, t, performance_data)
            
            # Traditional output formats
            if not args.hdf5_output:
                if output_mode == "binary_mask":
                    os.makedirs(save_base, exist_ok=False)
                    write_masks_to_folder(masks, save_base)
                else:
                    save_file = save_base + ".json"
                    with open(save_file, "w") as f:
                        json.dump(masks, f)
            
            if tracker:
                tracker.end_timer("output_saving")
            
            # Log per-image metrics to MLflow
            if mlflow_enabled and tracker:
                per_image_metrics = {
                    f"per_image_{k}": v for k, v in tracker.get_metrics().items()
                    if k in ["inference", "image_loading", "mask_saving", "visualization"]
                }
                per_image_metrics.update({
                    "masks_generated": num_masks,
                    "masks_per_second": num_masks / tracker.metrics.get("inference", {}).get("duration_seconds", 1)
                })
                log_to_mlflow(per_image_metrics, image_info)
        
        # End total processing timer
        if tracker:
            tracker.end_timer("total_processing")
        
        # Print summary statistics
        if args.track_performance:
            print("\n" + "="*50)
            print("PERFORMANCE SUMMARY")
            print("="*50)
            print(f"Total images processed: {total_images}")
            print(f"Total masks generated: {total_masks}")
            if total_images > 0:
                print(f"Average masks per image: {total_masks/total_images:.1f}")
                if total_inference_time > 0:
                    print(f"Total inference time: {total_inference_time:.4f}s")
                    print(f"Average inference time per image: {total_inference_time/total_images:.4f}s")
                    print(f"Images per second: {total_images/total_inference_time:.2f}")
                    print(f"Masks per second: {total_masks/total_inference_time:.2f}")
            
            if tracker:
                print("\nDetailed Performance Metrics:")
                tracker.log_metrics("  ")
                
                # Log summary metrics to MLflow
                if mlflow_enabled:
                    summary_metrics = {
                        "total_images": total_images,
                        "total_masks": total_masks,
                        "avg_masks_per_image": total_masks/total_images if total_images > 0 else 0,
                        "total_inference_time": total_inference_time,
                        "avg_inference_time_per_image": total_inference_time/total_images if total_images > 0 else 0,
                        "images_per_second": total_images/total_inference_time if total_inference_time > 0 else 0,
                        "masks_per_second": total_masks/total_inference_time if total_inference_time > 0 else 0
                    }
                    summary_metrics.update(tracker.get_metrics())
                    log_to_mlflow(summary_metrics)
                
    finally:
        # End MLflow run
        if mlflow_enabled:
            mlflow.end_run()
                
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
