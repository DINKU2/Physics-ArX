#!/usr/bin/env python3
"""
Visualize patient results: Input CBCT, Ground Truth CT, Output sCT, and Output RTSTRUCTS

Usage:
    python visualize_patient.py --patient 0
    python visualize_patient.py --patient 0 --slice 64
    python visualize_patient.py --patient 0 --all_slices
    python visualize_patient.py --patient 0 --interactive  # Interactive scrolling with slider
"""

import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from pathlib import Path


def load_input_data(patient_idx, dataroot='./datasets/psAAPM', phase='test'):
    """Load input CBCT and ground truth CT from npz file"""
    data_dir = Path(dataroot) / phase
    npz_files = sorted(list(data_dir.glob('*.npz')))
    
    if patient_idx >= len(npz_files):
        raise ValueError(f"Patient index {patient_idx} out of range. Only {len(npz_files)} patients available.")
    
    npz_path = npz_files[patient_idx]
    data = np.load(npz_path)
    
    cbct = data['CBCT']  # Input CBCT image
    ct_gt = data['CT']   # Ground truth CT image
    patient_name = npz_path.stem
    
    print(f"Loaded input: {npz_path}")
    print(f"  CBCT shape: {cbct.shape}, range: [{cbct.min():.2f}, {cbct.max():.2f}]")
    print(f"  CT (GT) shape: {ct_gt.shape}, range: [{ct_gt.min():.2f}, {ct_gt.max():.2f}]")
    
    return cbct, ct_gt, patient_name


def load_output_data(patient_name, results_dir='./results/msk_aapm_stabilized_eso4/test_latest/npz_images'):
    """Load output sCT and RTSTRUCTS from nrrd files"""
    results_path = Path(results_dir)
    
    sct_path = results_path / f"{patient_name}_CBCT2CT.nrrd"
    rtstruct_path = results_path / f"{patient_name}_RTSTRUCTS.nrrd"
    
    if not sct_path.exists():
        raise FileNotFoundError(f"Output sCT not found: {sct_path}")
    if not rtstruct_path.exists():
        raise FileNotFoundError(f"Output RTSTRUCTS not found: {rtstruct_path}")
    
    sct_img = sitk.ReadImage(str(sct_path))
    rtstruct_img = sitk.ReadImage(str(rtstruct_path))
    
    sct = sitk.GetArrayFromImage(sct_img)
    rtstruct = sitk.GetArrayFromImage(rtstruct_img)
    
    print(f"Loaded outputs:")
    print(f"  sCT shape: {sct.shape}, range: [{sct.min():.2f}, {sct.max():.2f}]")
    print(f"  RTSTRUCTS shape: {rtstruct.shape}, range: [{rtstruct.min():.2f}, {rtstruct.max():.2f}]")
    print(f"  RTSTRUCTS unique labels: {np.unique(rtstruct)}")
    
    return sct, rtstruct


def visualize_slice(cbct, ct_gt, sct, rtstruct, slice_idx, patient_name, save_path=None):
    """Visualize a single slice from all four volumes: CBCT, Ground Truth CT, Output sCT, and RTSTRUCTS"""
    depth = cbct.shape[0]
    
    if slice_idx >= depth:
        slice_idx = depth // 2
        print(f"Slice index out of range, using middle slice: {slice_idx}")
    
    # Get slices (assuming DxHxW format)
    cbct_slice = cbct[slice_idx, :, :]
    ct_gt_slice = ct_gt[slice_idx, :, :]
    sct_slice = sct[slice_idx, :, :]
    rtstruct_slice = rtstruct[slice_idx, :, :]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Calculate HU statistics
    cbct_hu_min, cbct_hu_max = cbct.min(), cbct.max()
    cbct_hu_mean = cbct.mean()
    cbct_slice_min, cbct_slice_max = cbct_slice.min(), cbct_slice.max()
    cbct_slice_mean = cbct_slice.mean()
    
    ct_gt_hu_min, ct_gt_hu_max = ct_gt.min(), ct_gt.max()
    ct_gt_hu_mean = ct_gt.mean()
    ct_gt_slice_min, ct_gt_slice_max = ct_gt_slice.min(), ct_gt_slice.max()
    ct_gt_slice_mean = ct_gt_slice.mean()
    
    sct_hu_min, sct_hu_max = sct.min(), sct.max()
    sct_hu_mean = sct.mean()
    sct_slice_min, sct_slice_max = sct_slice.min(), sct_slice.max()
    sct_slice_mean = sct_slice.mean()
    
    # Input CBCT
    im1 = axes[0].imshow(cbct_slice, cmap='gray', vmin=cbct.min(), vmax=cbct.max())
    axes[0].set_title(f'Input CBCT\nSlice {slice_idx}/{depth-1}', fontsize=12, fontweight='bold')
    axes[0].text(0.5, -0.12, f'Slice: [{cbct_slice_min:.3f}, {cbct_slice_max:.3f}], Mean: {cbct_slice_mean:.3f}\n'
                             f'Volume: [{cbct_hu_min:.3f}, {cbct_hu_max:.3f}], Mean: {cbct_hu_mean:.3f}\n'
                             f'(Normalized 0-1)\n'
                             f'CBCT: Less accurate HU', 
                 transform=axes[0].transAxes, ha='center', fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Ground Truth CT
    im2 = axes[1].imshow(ct_gt_slice, cmap='gray', vmin=ct_gt.min(), vmax=ct_gt.max())
    axes[1].set_title(f'Ground Truth CT\nSlice {slice_idx}/{depth-1}', fontsize=12, fontweight='bold')
    axes[1].text(0.5, -0.12, f'Slice: [{ct_gt_slice_min:.3f}, {ct_gt_slice_max:.3f}], Mean: {ct_gt_slice_mean:.3f}\n'
                             f'Volume: [{ct_gt_hu_min:.3f}, {ct_gt_hu_max:.3f}], Mean: {ct_gt_hu_mean:.3f}\n'
                             f'(Normalized 0-1)\n'
                             f'CT: Accurate HU values', 
                 transform=axes[1].transAxes, ha='center', fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Output sCT
    im3 = axes[2].imshow(sct_slice, cmap='gray', vmin=sct.min(), vmax=sct.max())
    axes[2].set_title(f'Output sCT\nSlice {slice_idx}/{depth-1}', fontsize=12, fontweight='bold')
    axes[2].text(0.5, -0.12, f'Slice: [{sct_slice_min:.3f}, {sct_slice_max:.3f}], Mean: {sct_slice_mean:.3f}\n'
                             f'Volume: [{sct_hu_min:.3f}, {sct_hu_max:.3f}], Mean: {sct_hu_mean:.3f}\n'
                             f'(Normalized 0-1)\n'
                             f'CT: Accurate HU values', 
                 transform=axes[2].transAxes, ha='center', fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Output RTSTRUCTS (with overlay on sCT)
    axes[3].imshow(sct_slice, cmap='gray', vmin=sct.min(), vmax=sct.max())
    # Overlay RTSTRUCTS with different colors for different labels
    rtstruct_colored = np.zeros((*rtstruct_slice.shape, 3))
    unique_labels = np.unique(rtstruct_slice)
    colors = {
        0: [0, 0, 0],      # Background - transparent
        1: [1, 0, 0],      # Lungs - Red
        2: [0, 1, 0],      # Heart - Green
        3: [0, 0, 1],      # Spinal Cord - Blue
        4: [1, 1, 0],      # Esophagus - Yellow
    }
    for label in unique_labels:
        if label > 0:  # Skip background
            mask = rtstruct_slice == label
            color = colors.get(int(label), [1, 1, 1])  # Default white for unknown labels
            rtstruct_colored[mask] = color
    
    axes[3].imshow(rtstruct_colored, alpha=0.5, interpolation='nearest')
    axes[3].set_title(f'Output RTSTRUCTS\nSlice {slice_idx}/{depth-1}', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # Add legend for RTSTRUCTS
    legend_labels = []
    for label in sorted(unique_labels):
        if label > 0:
            label_names = {1: 'Lungs', 2: 'Heart', 3: 'Spinal Cord', 4: 'Esophagus'}
            legend_labels.append(f'Label {int(label)}: {label_names.get(int(label), "Unknown")}')
    
    if legend_labels:
        fig.text(0.5, 0.04, ' | '.join(legend_labels), ha='center', fontsize=10)
    
    # Add typical HU ranges reference at the bottom
    fig.text(0.5, 0.01, 'Note: Values are normalized (0-1). CBCT→CT translation improves HU accuracy. CT HU: Air=-1000, Lung=-500~-900, Soft tissue=-100~100, Bone=200~3000', 
             ha='center', fontsize=8, style='italic', color='gray')
    
    plt.suptitle(f'Patient: {patient_name}', fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(bottom=0.18)  # Make room for HU text and reference ranges
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_interactive(cbct, ct_gt, sct, rtstruct, patient_name):
    """Interactive visualization with scrollbar to navigate through all slices"""
    depth = cbct.shape[0]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(bottom=0.18)
    
    # Initial slice (middle)
    initial_slice = depth // 2
    
    # Get initial slices
    cbct_slice = cbct[initial_slice, :, :]
    ct_gt_slice = ct_gt[initial_slice, :, :]
    sct_slice = sct[initial_slice, :, :]
    rtstruct_slice = rtstruct[initial_slice, :, :]
    
    # Create RTSTRUCTS color overlay
    rtstruct_colored = np.zeros((*rtstruct_slice.shape, 3))
    unique_labels = np.unique(rtstruct)
    colors = {
        1: [1, 0, 0],      # Lungs - Red
        2: [0, 1, 0],      # Heart - Green
        3: [0, 0, 1],      # Spinal Cord - Blue
        4: [1, 1, 0],      # Esophagus - Yellow
    }
    # Populate initial overlay
    for label in unique_labels:
        if label > 0:
            mask = rtstruct_slice == label
            color = colors.get(int(label), [1, 1, 1])
            rtstruct_colored[mask] = color
    
    # Input CBCT
    im1 = axes[0].imshow(cbct_slice, cmap='gray', vmin=cbct.min(), vmax=cbct.max())
    cbct_hu_min, cbct_hu_max = cbct.min(), cbct.max()
    cbct_hu_mean = cbct.mean()
    cbct_slice_min, cbct_slice_max = cbct_slice.min(), cbct_slice.max()
    cbct_slice_mean = cbct_slice.mean()
    axes[0].set_title(f'Input CBCT\nSlice {initial_slice}/{depth-1}', fontsize=12, fontweight='bold')
    axes[0].text(0.5, -0.12, f'Slice: [{cbct_slice_min:.3f}, {cbct_slice_max:.3f}], Mean: {cbct_slice_mean:.3f}\n'
                             f'Volume: [{cbct_hu_min:.3f}, {cbct_hu_max:.3f}], Mean: {cbct_hu_mean:.3f}\n'
                             f'(Normalized 0-1)\n'
                             f'CBCT: Less accurate HU', 
                 transform=axes[0].transAxes, ha='center', fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].axis('off')
    
    # Ground Truth CT
    im2 = axes[1].imshow(ct_gt_slice, cmap='gray', vmin=ct_gt.min(), vmax=ct_gt.max())
    ct_gt_hu_min, ct_gt_hu_max = ct_gt.min(), ct_gt.max()
    ct_gt_hu_mean = ct_gt.mean()
    ct_gt_slice_min, ct_gt_slice_max = ct_gt_slice.min(), ct_gt_slice.max()
    ct_gt_slice_mean = ct_gt_slice.mean()
    axes[1].set_title(f'Ground Truth CT\nSlice {initial_slice}/{depth-1}', fontsize=12, fontweight='bold')
    axes[1].text(0.5, -0.12, f'Slice: [{ct_gt_slice_min:.3f}, {ct_gt_slice_max:.3f}], Mean: {ct_gt_slice_mean:.3f}\n'
                             f'Volume: [{ct_gt_hu_min:.3f}, {ct_gt_hu_max:.3f}], Mean: {ct_gt_hu_mean:.3f}\n'
                             f'(Normalized 0-1)\n'
                             f'CT: Accurate HU values', 
                 transform=axes[1].transAxes, ha='center', fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1].axis('off')
    
    # Output sCT
    im3 = axes[2].imshow(sct_slice, cmap='gray', vmin=sct.min(), vmax=sct.max())
    sct_hu_min, sct_hu_max = sct.min(), sct.max()
    sct_hu_mean = sct.mean()
    sct_slice_min, sct_slice_max = sct_slice.min(), sct_slice.max()
    sct_slice_mean = sct_slice.mean()
    axes[2].set_title(f'Output sCT\nSlice {initial_slice}/{depth-1}', fontsize=12, fontweight='bold')
    axes[2].text(0.5, -0.12, f'Slice: [{sct_slice_min:.3f}, {sct_slice_max:.3f}], Mean: {sct_slice_mean:.3f}\n'
                             f'Volume: [{sct_hu_min:.3f}, {sct_hu_max:.3f}], Mean: {sct_hu_mean:.3f}\n'
                             f'(Normalized 0-1)\n'
                             f'CT: Accurate HU values', 
                 transform=axes[2].transAxes, ha='center', fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[2].axis('off')
    
    # Output RTSTRUCTS (with overlay on sCT)
    im4_bg = axes[3].imshow(sct_slice, cmap='gray', vmin=sct.min(), vmax=sct.max())
    im4_overlay = axes[3].imshow(rtstruct_colored, alpha=0.5, interpolation='nearest')
    axes[3].set_title(f'Output RTSTRUCTS\nSlice {initial_slice}/{depth-1}', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # Add legend for RTSTRUCTS
    legend_labels = []
    for label in sorted(unique_labels):
        if label > 0:
            label_names = {1: 'Lungs', 2: 'Heart', 3: 'Spinal Cord', 4: 'Esophagus'}
            legend_labels.append(f'Label {int(label)}: {label_names.get(int(label), "Unknown")}')
    
    if legend_labels:
        fig.text(0.5, 0.10, ' | '.join(legend_labels), ha='center', fontsize=10)
    
    plt.suptitle(f'Patient: {patient_name} - Use slider to scroll through slices', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, depth-1, valinit=initial_slice, valstep=1, valfmt='%d')
    
    # Add typical HU ranges reference at the bottom
    fig.text(0.5, 0.05, 'Note: Values are normalized (0-1). CBCT→CT translation improves HU accuracy. CT HU: Air=-1000, Lung=-500~-900, Soft tissue=-100~100, Bone=200~3000', 
             ha='center', fontsize=8, style='italic', color='gray')
    
    def update_slice(val):
        """Update all four images when slider changes"""
        slice_idx = int(slider.val)
        
        # Get new slices
        cbct_slice = cbct[slice_idx, :, :]
        ct_gt_slice = ct_gt[slice_idx, :, :]
        sct_slice = sct[slice_idx, :, :]
        rtstruct_slice = rtstruct[slice_idx, :, :]
        
        # Calculate HU stats for current slice
        cbct_slice_min, cbct_slice_max = cbct_slice.min(), cbct_slice.max()
        cbct_slice_mean = cbct_slice.mean()
        ct_gt_slice_min, ct_gt_slice_max = ct_gt_slice.min(), ct_gt_slice.max()
        ct_gt_slice_mean = ct_gt_slice.mean()
        sct_slice_min, sct_slice_max = sct_slice.min(), sct_slice.max()
        sct_slice_mean = sct_slice.mean()
        
        # Update CBCT
        im1.set_array(cbct_slice)
        axes[0].set_title(f'Input CBCT\nSlice {slice_idx}/{depth-1}', fontsize=12, fontweight='bold')
        # Clear previous text and add new
        for txt in axes[0].texts:
            if txt.get_position()[1] < 0:  # Text below axis
                txt.remove()
        axes[0].text(0.5, -0.12, f'Slice: [{cbct_slice_min:.3f}, {cbct_slice_max:.3f}], Mean: {cbct_slice_mean:.3f}\n'
                                 f'(Normalized 0-1)\n'
                                 f'CBCT: Less accurate HU due to\n'
                                 f'scatter artifacts & noise', 
                     transform=axes[0].transAxes, ha='center', fontsize=7,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Update Ground Truth CT
        im2.set_array(ct_gt_slice)
        axes[1].set_title(f'Ground Truth CT\nSlice {slice_idx}/{depth-1}', fontsize=12, fontweight='bold')
        # Clear previous text and add new
        for txt in axes[1].texts:
            if txt.get_position()[1] < 0:  # Text below axis
                txt.remove()
        axes[1].text(0.5, -0.12, f'Slice: [{ct_gt_slice_min:.3f}, {ct_gt_slice_max:.3f}], Mean: {ct_gt_slice_mean:.3f}\n'
                                 f'(Normalized 0-1)\n'
                                 f'CT: Accurate HU values', 
                     transform=axes[1].transAxes, ha='center', fontsize=7,
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Update sCT
        im3.set_array(sct_slice)
        axes[2].set_title(f'Output sCT\nSlice {slice_idx}/{depth-1}', fontsize=12, fontweight='bold')
        # Clear previous text and add new
        for txt in axes[2].texts:
            if txt.get_position()[1] < 0:  # Text below axis
                txt.remove()
        axes[2].text(0.5, -0.12, f'Slice: [{sct_slice_min:.3f}, {sct_slice_max:.3f}], Mean: {sct_slice_mean:.3f}\n'
                                 f'(Normalized 0-1)\n'
                                 f'CT: Accurate HU values', 
                     transform=axes[2].transAxes, ha='center', fontsize=7,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Update RTSTRUCTS overlay
        rtstruct_colored = np.zeros((*rtstruct_slice.shape, 3))
        for label in unique_labels:
            if label > 0:
                mask = rtstruct_slice == label
                color = colors.get(int(label), [1, 1, 1])
                rtstruct_colored[mask] = color
        
        im4_bg.set_array(sct_slice)
        im4_overlay.set_array(rtstruct_colored)
        axes[3].set_title(f'Output RTSTRUCTS\nSlice {slice_idx}/{depth-1}', fontsize=12, fontweight='bold')
        
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    slider.on_changed(update_slice)
    
    # Add keyboard navigation
    def on_key(event):
        if event.key == 'right' or event.key == 'up':
            if slider.val < depth - 1:
                slider.set_val(slider.val + 1)
        elif event.key == 'left' or event.key == 'down':
            if slider.val > 0:
                slider.set_val(slider.val - 1)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print(f"\nInteractive mode: Use the slider or arrow keys (←/→ or ↑/↓) to navigate through slices")
    print(f"Total slices: {depth}")
    
    plt.show()


def visualize_all_slices(cbct, ct_gt, sct, rtstruct, patient_name, output_dir='./visualizations'):
    """Create a grid visualization of multiple slices"""
    depth = cbct.shape[0]
    n_slices = min(9, depth)  # Show up to 9 slices in a 3x3 grid
    slice_indices = np.linspace(0, depth-1, n_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, slice_num in enumerate(slice_indices):
        cbct_slice = cbct[slice_num, :, :]
        sct_slice = sct[slice_num, :, :]
        rtstruct_slice = rtstruct[slice_num, :, :]
        
        # Create overlay
        axes[idx].imshow(sct_slice, cmap='gray', vmin=sct.min(), vmax=sct.max())
        
        # Overlay RTSTRUCTS
        rtstruct_colored = np.zeros((*rtstruct_slice.shape, 3))
        unique_labels = np.unique(rtstruct_slice)
        colors = {
            1: [1, 0, 0],      # Lungs - Red
            2: [0, 1, 0],      # Heart - Green
            3: [0, 0, 1],      # Spinal Cord - Blue
            4: [1, 1, 0],      # Esophagus - Yellow
        }
        for label in unique_labels:
            if label > 0:
                mask = rtstruct_slice == label
                color = colors.get(int(label), [1, 1, 1])
                rtstruct_colored[mask] = color
        
        axes[idx].imshow(rtstruct_colored, alpha=0.5, interpolation='nearest')
        axes[idx].set_title(f'Slice {slice_num}', fontsize=10)
        axes[idx].axis('off')
    
    plt.suptitle(f'Patient: {patient_name} - All Slices (sCT + RTSTRUCTS overlay)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{patient_name}_all_slices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved all slices visualization to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize patient results')
    parser.add_argument('--patient', type=int, default=0, 
                       help='Patient index (0-based)')
    parser.add_argument('--slice', type=int, default=None,
                       help='Specific slice index to visualize (default: middle slice)')
    parser.add_argument('--all_slices', action='store_true',
                       help='Show grid of all slices')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode with scrollbar to navigate through all slices')
    parser.add_argument('--dataroot', type=str, default='./datasets/psAAPM',
                       help='Path to dataset directory')
    parser.add_argument('--results_dir', type=str, 
                       default='./results/msk_aapm_stabilized_eso4/test_latest/npz_images',
                       help='Path to results directory')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization (optional)')
    
    args = parser.parse_args()
    
    # Load input data (CBCT and ground truth CT)
    cbct, ct_gt, patient_name = load_input_data(args.patient, args.dataroot)
    
    # Load output data
    sct, rtstruct = load_output_data(patient_name, args.results_dir)
    
    # Verify shapes match
    if cbct.shape != sct.shape:
        print(f"Warning: Shape mismatch! CBCT: {cbct.shape}, sCT: {sct.shape}")
    if cbct.shape != ct_gt.shape:
        print(f"Warning: Shape mismatch! CBCT: {cbct.shape}, CT (GT): {ct_gt.shape}")
    
    # Visualize
    if args.interactive:
        visualize_interactive(cbct, ct_gt, sct, rtstruct, patient_name)
    elif args.all_slices:
        visualize_all_slices(cbct, ct_gt, sct, rtstruct, patient_name, args.save)
    else:
        slice_idx = args.slice if args.slice is not None else cbct.shape[0] // 2
        visualize_slice(cbct, ct_gt, sct, rtstruct, slice_idx, patient_name, args.save)


if __name__ == '__main__':
    main()
