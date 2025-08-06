#!/usr/bin/env python3
"""
Utility functions for loading and manipulating HDF5 SAM output files.

This module provides comprehensive tools for working with the structured HDF5 output
from the AMG script, including loading, filtering, statistics, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    raise ImportError("h5py is required. Install with: pip install h5py")


@dataclass
class MaskData:
    """Data class for individual mask information."""
    id: int
    segmentation: np.ndarray
    area: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    point_coords: Tuple[float, float]
    predicted_iou: float
    stability_score: float
    crop_box: Tuple[float, float, float, float]


class SAMResults:
    """
    Class for loading and manipulating SAM segmentation results from HDF5 files.
    
    Provides comprehensive functionality for:
    - Loading all data from HDF5 files
    - Filtering masks by various criteria  
    - Computing statistics
    - Creating visualizations
    - Exporting filtered results
    """
    
    def __init__(self, hdf5_path: str):
        """
        Load SAM results from HDF5 file.
        
        Args:
            hdf5_path: Path to the HDF5 file created by AMG script
        """
        self.hdf5_path = hdf5_path
        self._load_data()
    
    def _load_data(self):
        """Load all data from the HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load metadata
            self.metadata = {}
            meta_group = f['metadata']
            for key in meta_group.attrs:
                self.metadata[key] = meta_group.attrs[key]
            
            # Load performance metrics if available
            self.performance_metrics = {}
            if 'performance' in meta_group:
                perf_group = meta_group['performance']
                for metric_name in perf_group:
                    if isinstance(perf_group[metric_name], h5py.Group):
                        self.performance_metrics[metric_name] = {}
                        for sub_key in perf_group[metric_name].attrs:
                            self.performance_metrics[metric_name][sub_key] = perf_group[metric_name].attrs[sub_key]
                    else:
                        self.performance_metrics[metric_name] = perf_group.attrs[metric_name]
            
            # Load original image
            img_group = f['image']
            self.original_image = img_group['original'][:]
            self.image_attrs = dict(img_group['original'].attrs)
            
            # Load masks
            masks_group = f['masks']
            self.mask_segmentations = masks_group['segmentations'][:]
            self.mask_metadata_raw = masks_group['metadata'][:]
            
            # Load visualization if available
            self.visualization_image = None
            if 'visualization' in f:
                viz_group = f['visualization']
                if 'image' in viz_group:
                    self.visualization_image = viz_group['image'][:]
        
        # Convert structured array to list of MaskData objects
        self.masks = []
        for i, mask_meta in enumerate(self.mask_metadata_raw):
            mask_data = MaskData(
                id=int(mask_meta['id']),
                segmentation=self.mask_segmentations[i],
                area=float(mask_meta['area']) if 'area' in mask_meta.dtype.names else 0.0,
                bbox=(
                    float(mask_meta['bbox_x']) if 'bbox_x' in mask_meta.dtype.names else 0.0,
                    float(mask_meta['bbox_y']) if 'bbox_y' in mask_meta.dtype.names else 0.0,
                    float(mask_meta['bbox_w']) if 'bbox_w' in mask_meta.dtype.names else 0.0,
                    float(mask_meta['bbox_h']) if 'bbox_h' in mask_meta.dtype.names else 0.0
                ),
                point_coords=(
                    float(mask_meta['point_x']) if 'point_x' in mask_meta.dtype.names else 0.0,
                    float(mask_meta['point_y']) if 'point_y' in mask_meta.dtype.names else 0.0
                ),
                predicted_iou=float(mask_meta['predicted_iou']) if 'predicted_iou' in mask_meta.dtype.names else 0.0,
                stability_score=float(mask_meta['stability_score']) if 'stability_score' in mask_meta.dtype.names else 0.0,
                crop_box=(
                    float(mask_meta['crop_box_x']) if 'crop_box_x' in mask_meta.dtype.names else 0.0,
                    float(mask_meta['crop_box_y']) if 'crop_box_y' in mask_meta.dtype.names else 0.0,
                    float(mask_meta['crop_box_w']) if 'crop_box_w' in mask_meta.dtype.names else 0.0,
                    float(mask_meta['crop_box_h']) if 'crop_box_h' in mask_meta.dtype.names else 0.0
                )
            )
            self.masks.append(mask_data)
    
    def get_info(self) -> Dict[str, Any]:
        """Get summary information about the segmentation results."""
        return {
            'num_masks': len(self.masks),
            'image_shape': self.original_image.shape,
            'creation_time': self.metadata.get('creation_time', 'Unknown'),
            'model_type': self.metadata.get('model_type', 'Unknown'),
            'device': self.metadata.get('device', 'Unknown'),
            'image_path': self.metadata.get('image_path', 'Unknown'),
            'has_visualization': self.visualization_image is not None,
            'has_performance_metrics': bool(self.performance_metrics)
        }
    
    def filter_masks(
        self,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        min_iou: Optional[float] = None,
        min_stability: Optional[float] = None,
        bbox_filter: Optional[Tuple[int, int, int, int]] = None,  # x, y, w, h
        top_n: Optional[int] = None,
        sort_by: str = 'area'
    ) -> List[MaskData]:
        """
        Filter masks based on various criteria.
        
        Args:
            min_area: Minimum mask area in pixels
            max_area: Maximum mask area in pixels  
            min_iou: Minimum predicted IoU score
            min_stability: Minimum stability score
            bbox_filter: Only include masks that intersect with this bounding box (x, y, w, h)
            top_n: Return only the top N masks (after other filtering)
            sort_by: Sort criterion ('area', 'predicted_iou', 'stability_score')
            
        Returns:
            List of filtered MaskData objects
        """
        filtered_masks = list(self.masks)
        
        # Apply filters
        if min_area is not None:
            filtered_masks = [m for m in filtered_masks if m.area >= min_area]
        
        if max_area is not None:
            filtered_masks = [m for m in filtered_masks if m.area <= max_area]
        
        if min_iou is not None:
            filtered_masks = [m for m in filtered_masks if m.predicted_iou >= min_iou]
        
        if min_stability is not None:
            filtered_masks = [m for m in filtered_masks if m.stability_score >= min_stability]
        
        if bbox_filter is not None:
            x_filter, y_filter, w_filter, h_filter = bbox_filter
            filtered_masks = [
                m for m in filtered_masks 
                if self._bbox_intersects(m.bbox, (x_filter, y_filter, w_filter, h_filter))
            ]
        
        # Sort masks
        if sort_by == 'area':
            filtered_masks.sort(key=lambda m: m.area, reverse=True)
        elif sort_by == 'predicted_iou':
            filtered_masks.sort(key=lambda m: m.predicted_iou, reverse=True)
        elif sort_by == 'stability_score':
            filtered_masks.sort(key=lambda m: m.stability_score, reverse=True)
        
        # Take top N
        if top_n is not None:
            filtered_masks = filtered_masks[:top_n]
        
        return filtered_masks
    
    def _bbox_intersects(self, bbox1: Tuple[float, float, float, float], 
                        bbox2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes intersect."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def get_statistics(self, masks: Optional[List[MaskData]] = None) -> Dict[str, Any]:
        """
        Compute statistics for the masks.
        
        Args:
            masks: List of masks to analyze. If None, uses all masks.
            
        Returns:
            Dictionary with statistical information
        """
        if masks is None:
            masks = self.masks
        
        if not masks:
            return {'num_masks': 0}
        
        areas = [m.area for m in masks]
        ious = [m.predicted_iou for m in masks if m.predicted_iou > 0]
        stabilities = [m.stability_score for m in masks if m.stability_score > 0]
        
        stats = {
            'num_masks': len(masks),
            'area_stats': {
                'mean': np.mean(areas),
                'median': np.median(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas),
                'total': np.sum(areas)
            }
        }
        
        if ious:
            stats['iou_stats'] = {
                'mean': np.mean(ious),
                'median': np.median(ious),
                'std': np.std(ious),
                'min': np.min(ious),
                'max': np.max(ious)
            }
        
        if stabilities:
            stats['stability_stats'] = {
                'mean': np.mean(stabilities),
                'median': np.median(stabilities),
                'std': np.std(stabilities),
                'min': np.min(stabilities),
                'max': np.max(stabilities)
            }
        
        return stats
    
    def visualize_masks(
        self,
        masks: Optional[List[MaskData]] = None,
        random_colors: bool = True,
        show_contours: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Create a visualization of the masks overlaid on the original image.
        
        Args:
            masks: List of masks to visualize. If None, uses all masks.
            random_colors: Use random colors for masks
            show_contours: Show contour lines around masks
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure object
        """
        if masks is None:
            masks = self.masks
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(self.original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Image with masks
        img_with_masks = self.original_image.copy()
        
        if masks:
            # Create overlay
            overlay = np.zeros((*self.original_image.shape[:2], 4))
            
            for i, mask in enumerate(masks):
                if random_colors:
                    color = np.random.rand(3)
                else:
                    color = np.array([30/255, 144/255, 255/255])
                
                mask_pixels = mask.segmentation > 0
                overlay[mask_pixels] = [*color, 0.6]
                
                # Add contours if requested
                if show_contours:
                    contours, _ = cv2.findContours(
                        mask.segmentation.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(img_with_masks, contours, -1, (255, 255, 255), 2)
            
            # Blend overlay with original image
            img_with_masks = img_with_masks.astype(float)
            for c in range(3):
                img_with_masks[:, :, c] = (
                    img_with_masks[:, :, c] * (1 - overlay[:, :, 3]) + 
                    overlay[:, :, c] * 255 * overlay[:, :, 3]
                )
            img_with_masks = img_with_masks.astype(np.uint8)
        
        axes[1].imshow(img_with_masks)
        axes[1].set_title(f'Masks Overlay ({len(masks)} masks)')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_mask_grid(
        self,
        masks: Optional[List[MaskData]] = None,
        max_masks: int = 16,
        figsize: Tuple[int, int] = (16, 16)
    ) -> plt.Figure:
        """
        Create a grid showing individual masks.
        
        Args:
            masks: List of masks to show. If None, uses all masks.
            max_masks: Maximum number of masks to show
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure object
        """
        if masks is None:
            masks = self.masks
        
        masks = masks[:max_masks]
        n_masks = len(masks)
        
        if n_masks == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No masks to display', ha='center', va='center')
            ax.axis('off')
            return fig
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(n_masks)))
        rows = int(np.ceil(n_masks / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, mask in enumerate(masks):
            row = i // cols
            col = i % cols
            
            # Show mask on original image
            masked_img = self.original_image.copy()
            mask_pixels = mask.segmentation > 0
            masked_img[~mask_pixels] = masked_img[~mask_pixels] * 0.3
            
            axes[row, col].imshow(masked_img)
            axes[row, col].set_title(
                f'Mask {mask.id}\nArea: {mask.area:.0f}\nIoU: {mask.predicted_iou:.3f}',
                fontsize=8
            )
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_masks, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def export_filtered_masks(
        self,
        output_path: str,
        masks: Optional[List[MaskData]] = None,
        format: str = 'png'
    ):
        """
        Export filtered masks to individual files.
        
        Args:
            output_path: Directory to save the masks
            masks: List of masks to export. If None, uses all masks.
            format: Export format ('png', 'jpg', 'numpy')
        """
        import os
        if masks is None:
            masks = self.masks
        
        os.makedirs(output_path, exist_ok=True)
        
        for mask in masks:
            filename = f"mask_{mask.id:04d}"
            
            if format == 'png':
                Image.fromarray((mask.segmentation * 255).astype(np.uint8)).save(
                    os.path.join(output_path, f"{filename}.png")
                )
            elif format == 'jpg':
                Image.fromarray((mask.segmentation * 255).astype(np.uint8)).save(
                    os.path.join(output_path, f"{filename}.jpg")
                )
            elif format == 'numpy':
                np.save(os.path.join(output_path, f"{filename}.npy"), mask.segmentation)
        
        # Export metadata as CSV
        metadata_rows = []
        for mask in masks:
            metadata_rows.append({
                'id': mask.id,
                'area': mask.area,
                'bbox_x': mask.bbox[0],
                'bbox_y': mask.bbox[1],
                'bbox_w': mask.bbox[2],
                'bbox_h': mask.bbox[3],
                'point_x': mask.point_coords[0],
                'point_y': mask.point_coords[1],
                'predicted_iou': mask.predicted_iou,
                'stability_score': mask.stability_score,
                'crop_box_x': mask.crop_box[0],
                'crop_box_y': mask.crop_box[1],
                'crop_box_w': mask.crop_box[2],
                'crop_box_h': mask.crop_box[3]
            })
        
        pd.DataFrame(metadata_rows).to_csv(
            os.path.join(output_path, 'filtered_metadata.csv'), 
            index=False
        )
    
    def to_pandas(self, masks: Optional[List[MaskData]] = None) -> pd.DataFrame:
        """
        Convert mask metadata to a pandas DataFrame for analysis.
        
        Args:
            masks: List of masks to convert. If None, uses all masks.
            
        Returns:
            Pandas DataFrame with mask metadata
        """
        if masks is None:
            masks = self.masks
        
        data = []
        for mask in masks:
            data.append({
                'id': mask.id,
                'area': mask.area,
                'bbox_x': mask.bbox[0],
                'bbox_y': mask.bbox[1],
                'bbox_w': mask.bbox[2],
                'bbox_h': mask.bbox[3],
                'point_x': mask.point_coords[0],
                'point_y': mask.point_coords[1],
                'predicted_iou': mask.predicted_iou,
                'stability_score': mask.stability_score,
                'crop_box_x': mask.crop_box[0],
                'crop_box_y': mask.crop_box[1],
                'crop_box_w': mask.crop_box[2],
                'crop_box_h': mask.crop_box[3]
            })
        
        return pd.DataFrame(data)


def load_sam_results(hdf5_path: str) -> SAMResults:
    """
    Convenience function to load SAM results from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        SAMResults object
    """
    return SAMResults(hdf5_path)


def compare_results(results1: SAMResults, results2: SAMResults) -> Dict[str, Any]:
    """
    Compare two SAM result sets and return comparison statistics.
    
    Args:
        results1: First SAM results
        results2: Second SAM results
        
    Returns:
        Dictionary with comparison statistics
    """
    stats1 = results1.get_statistics()
    stats2 = results2.get_statistics()
    
    comparison = {
        'masks_difference': stats2['num_masks'] - stats1['num_masks'],
        'area_stats_comparison': {
            'mean_diff': stats2['area_stats']['mean'] - stats1['area_stats']['mean'],
            'total_diff': stats2['area_stats']['total'] - stats1['area_stats']['total']
        }
    }
    
    if 'iou_stats' in stats1 and 'iou_stats' in stats2:
        comparison['iou_comparison'] = {
            'mean_diff': stats2['iou_stats']['mean'] - stats1['iou_stats']['mean']
        }
    
    return comparison


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sam_hdf5_utils.py <path_to_hdf5_file>")
        sys.exit(1)
    
    # Load results
    results = load_sam_results(sys.argv[1])
    
    # Print basic info
    info = results.get_info()
    print("SAM Results Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Print statistics
    stats = results.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Example filtering
    large_masks = results.filter_masks(min_area=1000, min_iou=0.8, top_n=10)
    print(f"\nFound {len(large_masks)} large, high-quality masks")
    
    # Create visualization
    fig = results.visualize_masks(large_masks)
    plt.savefig('filtered_masks.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'filtered_masks.png'") 