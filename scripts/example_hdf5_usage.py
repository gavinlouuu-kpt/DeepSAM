#!/usr/bin/env python3
"""
Example script demonstrating HDF5 SAM results analysis.

This script shows how to:
1. Run AMG with HDF5 output
2. Load and analyze the results
3. Filter masks by various criteria
4. Create visualizations
5. Export filtered results
"""

import os
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Add the scripts directory to path to import our utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sam_hdf5_utils import load_sam_results


def run_amg_with_hdf5(image_path, output_dir, model_type="vit_t", device="cuda"):
    """
    Run AMG script with HDF5 output.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output
        model_type: SAM model type
        device: Device to run on
    """
    print(f"Running AMG on {image_path}...")
    
    cmd = [
        "python", "amg.py",
        "--input", image_path,
        "--output", output_dir,
        "--model-type", model_type,
        "--device", device,
        "--hdf5-output",
        "--track-performance"
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        print("AMG completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"AMG failed with error: {e}")
        return False


def analyze_results(hdf5_path):
    """
    Comprehensive analysis of SAM results.
    
    Args:
        hdf5_path: Path to HDF5 results file
    """
    print(f"\nAnalyzing results from {hdf5_path}...")
    
    # Load results
    results = load_sam_results(hdf5_path)
    
    # Basic information
    info = results.get_info()
    print("\n" + "="*50)
    print("BASIC INFORMATION")
    print("="*50)
    for key, value in info.items():
        print(f"{key:20}: {value}")
    
    # Overall statistics
    stats = results.get_statistics()
    print("\n" + "="*50)
    print("OVERALL STATISTICS")
    print("="*50)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, float):
                    print(f"  {subkey:15}: {subvalue:.3f}")
                else:
                    print(f"  {subkey:15}: {subvalue}")
        else:
            print(f"{key:20}: {value}")
    
    # Performance metrics
    if results.performance_metrics:
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        for metric_name, metric_data in results.performance_metrics.items():
            if isinstance(metric_data, dict):
                print(f"{metric_name}:")
                for sub_key, value in metric_data.items():
                    if isinstance(value, float):
                        print(f"  {sub_key:20}: {value:.4f}")
                    else:
                        print(f"  {sub_key:20}: {value}")
            else:
                print(f"{metric_name:25}: {metric_data}")
    
    return results


def demonstrate_filtering(results):
    """
    Demonstrate various filtering capabilities.
    
    Args:
        results: SAMResults object
    """
    print("\n" + "="*50)
    print("FILTERING DEMONSTRATIONS")
    print("="*50)
    
    # Filter 1: Large masks
    large_masks = results.filter_masks(min_area=1000, sort_by='area', top_n=20)
    print(f"Large masks (area >= 1000): {len(large_masks)}")
    
    # Filter 2: High-quality masks
    high_quality = results.filter_masks(min_iou=0.8, min_stability=0.9, sort_by='predicted_iou')
    print(f"High-quality masks (IoU >= 0.8, stability >= 0.9): {len(high_quality)}")
    
    # Filter 3: Medium-sized masks in center region
    h, w = results.original_image.shape[:2]
    center_region = (w//4, h//4, w//2, h//2)  # Center quarter of image
    center_masks = results.filter_masks(
        min_area=500, max_area=5000,
        bbox_filter=center_region,
        sort_by='area'
    )
    print(f"Medium masks in center region: {len(center_masks)}")
    
    # Filter 4: Top masks by different criteria
    top_by_area = results.filter_masks(sort_by='area', top_n=10)
    top_by_iou = results.filter_masks(sort_by='predicted_iou', top_n=10)
    top_by_stability = results.filter_masks(sort_by='stability_score', top_n=10)
    
    print(f"Top 10 by area: largest area = {top_by_area[0].area:.0f}")
    print(f"Top 10 by IoU: highest IoU = {top_by_iou[0].predicted_iou:.3f}")
    print(f"Top 10 by stability: highest stability = {top_by_stability[0].stability_score:.3f}")
    
    return {
        'large_masks': large_masks,
        'high_quality': high_quality,
        'center_masks': center_masks,
        'top_by_area': top_by_area,
        'top_by_iou': top_by_iou,
        'top_by_stability': top_by_stability
    }


def create_visualizations(results, filtered_sets, output_dir):
    """
    Create various visualizations.
    
    Args:
        results: SAMResults object
        filtered_sets: Dictionary of filtered mask sets
        output_dir: Directory to save visualizations
    """
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. All masks overview
    fig = results.visualize_masks(figsize=(15, 10))
    fig.suptitle(f'All Masks ({len(results.masks)} total)', fontsize=16)
    fig.savefig(os.path.join(viz_dir, "all_masks.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: all_masks.png")
    
    # 2. Large masks
    if filtered_sets['large_masks']:
        fig = results.visualize_masks(filtered_sets['large_masks'], figsize=(15, 10))
        fig.suptitle(f'Large Masks ({len(filtered_sets["large_masks"])} masks)', fontsize=16)
        fig.savefig(os.path.join(viz_dir, "large_masks.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved: large_masks.png")
    
    # 3. High-quality masks
    if filtered_sets['high_quality']:
        fig = results.visualize_masks(filtered_sets['high_quality'], figsize=(15, 10))
        fig.suptitle(f'High-Quality Masks ({len(filtered_sets["high_quality"])} masks)', fontsize=16)
        fig.savefig(os.path.join(viz_dir, "high_quality_masks.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved: high_quality_masks.png")
    
    # 4. Mask grid for top masks by area
    if filtered_sets['top_by_area']:
        fig = results.create_mask_grid(filtered_sets['top_by_area'], max_masks=16, figsize=(16, 16))
        fig.suptitle('Top 16 Masks by Area', fontsize=16)
        fig.savefig(os.path.join(viz_dir, "top_masks_grid.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved: top_masks_grid.png")
    
    # 5. Statistics comparison plot
    create_statistics_plot(results, filtered_sets, viz_dir)
    print("Saved: statistics_comparison.png")


def create_statistics_plot(results, filtered_sets, output_dir):
    """Create a statistics comparison plot."""
    datasets = {
        'All Masks': results.masks,
        'Large Masks': filtered_sets['large_masks'],
        'High Quality': filtered_sets['high_quality'],
        'Center Region': filtered_sets['center_masks']
    }
    
    # Collect statistics
    stats_data = {}
    for name, masks in datasets.items():
        if masks:
            stats = results.get_statistics(masks)
            stats_data[name] = {
                'count': stats['num_masks'],
                'mean_area': stats['area_stats']['mean'],
                'total_area': stats['area_stats']['total'],
                'mean_iou': stats.get('iou_stats', {}).get('mean', 0),
                'mean_stability': stats.get('stability_stats', {}).get('mean', 0)
            }
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    names = list(stats_data.keys())
    
    # Plot 1: Number of masks
    counts = [stats_data[name]['count'] for name in names]
    axes[0].bar(names, counts, color='skyblue')
    axes[0].set_title('Number of Masks')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Mean area
    mean_areas = [stats_data[name]['mean_area'] for name in names]
    axes[1].bar(names, mean_areas, color='lightgreen')
    axes[1].set_title('Mean Mask Area')
    axes[1].set_ylabel('Area (pixels)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Total area
    total_areas = [stats_data[name]['total_area'] for name in names]
    axes[2].bar(names, total_areas, color='lightcoral')
    axes[2].set_title('Total Mask Area')
    axes[2].set_ylabel('Total Area (pixels)')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Mean IoU
    mean_ious = [stats_data[name]['mean_iou'] for name in names]
    axes[3].bar(names, mean_ious, color='gold')
    axes[3].set_title('Mean Predicted IoU')
    axes[3].set_ylabel('IoU Score')
    axes[3].tick_params(axis='x', rotation=45)
    
    # Plot 5: Mean stability
    mean_stabilities = [stats_data[name]['mean_stability'] for name in names]
    axes[4].bar(names, mean_stabilities, color='plum')
    axes[4].set_title('Mean Stability Score')
    axes[4].set_ylabel('Stability Score')
    axes[4].tick_params(axis='x', rotation=45)
    
    # Hide the last subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def export_filtered_results(results, filtered_sets, output_dir):
    """
    Export filtered results to various formats.
    
    Args:
        results: SAMResults object
        filtered_sets: Dictionary of filtered mask sets
        output_dir: Directory to save exports
    """
    print("\n" + "="*50)
    print("EXPORTING FILTERED RESULTS")
    print("="*50)
    
    exports_dir = os.path.join(output_dir, "exports")
    os.makedirs(exports_dir, exist_ok=True)
    
    # Export high-quality masks as PNGs
    if filtered_sets['high_quality']:
        hq_dir = os.path.join(exports_dir, "high_quality_masks")
        results.export_filtered_masks(hq_dir, filtered_sets['high_quality'], format='png')
        print(f"Exported {len(filtered_sets['high_quality'])} high-quality masks to {hq_dir}")
    
    # Export large masks as numpy arrays
    if filtered_sets['large_masks']:
        large_dir = os.path.join(exports_dir, "large_masks_numpy")
        results.export_filtered_masks(large_dir, filtered_sets['large_masks'], format='numpy')
        print(f"Exported {len(filtered_sets['large_masks'])} large masks as numpy arrays to {large_dir}")
    
    # Export metadata as pandas DataFrame/CSV
    df_all = results.to_pandas()
    df_all.to_csv(os.path.join(exports_dir, "all_masks_metadata.csv"), index=False)
    print(f"Exported metadata for all {len(df_all)} masks to all_masks_metadata.csv")
    
    if filtered_sets['high_quality']:
        df_hq = results.to_pandas(filtered_sets['high_quality'])
        df_hq.to_csv(os.path.join(exports_dir, "high_quality_metadata.csv"), index=False)
        print(f"Exported metadata for {len(df_hq)} high-quality masks to high_quality_metadata.csv")


def main():
    """Main demonstration function."""
    if len(sys.argv) < 2:
        print("Usage: python example_hdf5_usage.py <input_image_path> [output_dir]")
        print("\nThis script will:")
        print("1. Run AMG with HDF5 output on the input image")
        print("2. Load and analyze the results")
        print("3. Demonstrate filtering capabilities")
        print("4. Create various visualizations")
        print("5. Export filtered results")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "hdf5_demo_output"
    
    if not os.path.exists(input_image):
        print(f"Error: Input image {input_image} does not exist")
        sys.exit(1)
    
    print("="*60)
    print("HDF5 SAM ANALYSIS DEMONSTRATION")
    print("="*60)
    print(f"Input image: {input_image}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Run AMG with HDF5 output
    if not run_amg_with_hdf5(input_image, output_dir):
        print("Failed to run AMG. Exiting.")
        sys.exit(1)
    
    # Find the generated HDF5 file
    image_name = os.path.splitext(os.path.basename(input_image))[0]
    hdf5_path = os.path.join(output_dir, f"{image_name}.h5")
    
    if not os.path.exists(hdf5_path):
        print(f"Error: Expected HDF5 file {hdf5_path} was not created")
        sys.exit(1)
    
    # Step 2: Load and analyze results
    results = analyze_results(hdf5_path)
    
    # Step 3: Demonstrate filtering
    filtered_sets = demonstrate_filtering(results)
    
    # Step 4: Create visualizations
    create_visualizations(results, filtered_sets, output_dir)
    
    # Step 5: Export filtered results
    export_filtered_results(results, filtered_sets, output_dir)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"HDF5 file: {hdf5_path}")
    print(f"Visualizations: {os.path.join(output_dir, 'visualizations')}")
    print(f"Exports: {os.path.join(output_dir, 'exports')}")
    
    # Final summary
    print(f"\nSummary:")
    print(f"- Total masks: {len(results.masks)}")
    print(f"- Large masks: {len(filtered_sets['large_masks'])}")
    print(f"- High-quality masks: {len(filtered_sets['high_quality'])}")
    print(f"- Center region masks: {len(filtered_sets['center_masks'])}")


if __name__ == "__main__":
    main() 