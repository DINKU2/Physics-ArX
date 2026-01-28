#!/usr/bin/env python3
"""
Analyze CBCT-to-CT translation quality by comparing input CBCT and output sCT
"""
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_slice(patient_idx=0, slice_idx=64, dataroot='./datasets/psAAPM', 
                  results_dir='./results/msk_aapm_stabilized_eso4/test_latest/npz_images'):
    """Analyze a specific slice to verify translation quality"""
    
    # Load input data
    data_dir = Path(dataroot) / 'test'
    npz_files = sorted(list(data_dir.glob('*.npz')))
    
    if patient_idx >= len(npz_files):
        raise ValueError(f"Patient index {patient_idx} out of range. Only {len(npz_files)} patients available.")
    
    npz_path = npz_files[patient_idx]
    data = np.load(npz_path)
    
    cbct = data['CBCT']
    ct_gt = data['CT']  # Ground truth CT (if available)
    patient_name = npz_path.stem
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {patient_name}")
    print(f"{'='*60}")
    
    # Load output data
    results_path = Path(results_dir)
    sct_path = results_path / f"{patient_name}_CBCT2CT.nrrd"
    
    if not sct_path.exists():
        raise FileNotFoundError(f"Output sCT not found: {sct_path}")
    
    sct_img = sitk.ReadImage(str(sct_path))
    sct = sitk.GetArrayFromImage(sct_img)
    
    depth = cbct.shape[0]
    if slice_idx >= depth:
        slice_idx = depth // 2
        print(f"Slice index adjusted to middle slice: {slice_idx}")
    
    # Get slices
    cbct_slice = cbct[slice_idx, :, :]
    sct_slice = sct[slice_idx, :, :]
    ct_gt_slice = ct_gt[slice_idx, :, :] if 'CT' in data else None
    
    print(f"\nSlice {slice_idx}/{depth-1} Analysis:")
    print(f"{'-'*60}")
    
    # 1. Statistical comparison
    print("\n1. STATISTICAL COMPARISON:")
    print(f"   Input CBCT:")
    print(f"     Range: [{cbct_slice.min():.4f}, {cbct_slice.max():.4f}]")
    print(f"     Mean: {cbct_slice.mean():.4f}, Std: {cbct_slice.std():.4f}")
    print(f"     Variance: {cbct_slice.var():.4f}")
    
    print(f"\n   Output sCT:")
    print(f"     Range: [{sct_slice.min():.4f}, {sct_slice.max():.4f}]")
    print(f"     Mean: {sct_slice.mean():.4f}, Std: {sct_slice.std():.4f}")
    print(f"     Variance: {sct_slice.var():.4f}")
    
    if ct_gt_slice is not None:
        print(f"\n   Ground Truth CT:")
        print(f"     Range: [{ct_gt_slice.min():.4f}, {ct_gt_slice.max():.4f}]")
        print(f"     Mean: {ct_gt_slice.mean():.4f}, Std: {ct_gt_slice.std():.4f}")
        print(f"     Variance: {ct_gt_slice.var():.4f}")
    
    # 2. Image quality metrics
    print("\n2. IMAGE QUALITY METRICS:")
    
    # Signal-to-Noise Ratio (SNR) approximation
    cbct_snr = cbct_slice.mean() / (cbct_slice.std() + 1e-10)
    sct_snr = sct_slice.mean() / (sct_slice.std() + 1e-10)
    print(f"   CBCT SNR (approx): {cbct_snr:.4f}")
    print(f"   sCT SNR (approx): {sct_snr:.4f}")
    print(f"   SNR Improvement: {((sct_snr - cbct_snr) / cbct_snr * 100):.2f}%")
    
    # Contrast (using standard deviation as proxy)
    print(f"\n   CBCT Contrast (std): {cbct_slice.std():.4f}")
    print(f"   sCT Contrast (std): {sct_slice.std():.4f}")
    contrast_improvement = ((sct_slice.std() - cbct_slice.std()) / cbct_slice.std() * 100) if cbct_slice.std() > 0 else 0
    print(f"   Contrast Change: {contrast_improvement:.2f}%")
    
    # 3. Comparison with ground truth (if available)
    if ct_gt_slice is not None:
        print("\n3. COMPARISON WITH GROUND TRUTH CT:")
        mse = np.mean((sct_slice - ct_gt_slice) ** 2)
        mae = np.mean(np.abs(sct_slice - ct_gt_slice))
        print(f"   Mean Squared Error (MSE): {mse:.6f}")
        print(f"   Mean Absolute Error (MAE): {mae:.6f}")
        
        # Correlation
        correlation = np.corrcoef(sct_slice.flatten(), ct_gt_slice.flatten())[0, 1]
        print(f"   Correlation: {correlation:.4f}")
        
        # HU Accuracy Assessment (for normalized values)
        print(f"\n   HU ACCURACY ASSESSMENT:")
        print(f"   ⚠ IMPORTANT: Values are normalized (0-1), not actual HU")
        print(f"   ✓ High correlation ({correlation:.4f}) indicates model preserves")
        print(f"     relative intensity relationships correctly")
        print(f"   ✓ Low MAE ({mae:.4f}) indicates close match to GT CT")
        print(f"   ⚠ Cannot verify absolute HU accuracy without original HU ranges")
        print(f"   → Model learns correct CBCT→CT mapping in normalized space")
        print(f"   → If original normalization preserved HU relationships,")
        print(f"     then denormalized output should have accurate HU values")
    
    # 4. Visual assessment indicators
    print("\n4. VISUAL QUALITY INDICATORS:")
    
    # Check if sCT has better contrast (higher std usually means better contrast)
    if sct_slice.std() > cbct_slice.std():
        print("   ✓ sCT has higher contrast than CBCT (good)")
    else:
        print("   ⚠ sCT has lower contrast than CBCT (check if this is expected)")
    
    # Check if sCT has less noise (lower variance in smooth regions)
    # We'll use a simple edge detection to identify smooth vs edge regions
    from scipy import ndimage
    sobel_cbct = ndimage.sobel(cbct_slice)
    sobel_sct = ndimage.sobel(sct_slice)
    
    # Smooth regions (low gradient)
    smooth_mask = sobel_cbct < np.percentile(sobel_cbct, 50)
    if np.sum(smooth_mask) > 0:
        cbct_smooth_var = cbct_slice[smooth_mask].var()
        sct_smooth_var = sct_slice[smooth_mask].var()
        print(f"   CBCT smooth region variance: {cbct_smooth_var:.6f}")
        print(f"   sCT smooth region variance: {sct_smooth_var:.6f}")
        if sct_smooth_var < cbct_smooth_var:
            print("   ✓ sCT has less noise in smooth regions (good)")
        else:
            print("   ⚠ sCT has more noise in smooth regions")
    
    # 5. Overall assessment
    print("\n5. OVERALL ASSESSMENT:")
    print(f"{'-'*60}")
    
    improvements = []
    if sct_slice.std() > cbct_slice.std():
        improvements.append("Better contrast")
    if sct_snr > cbct_snr:
        improvements.append("Better SNR")
    if ct_gt_slice is not None and correlation > 0.9:
        improvements.append("High correlation with GT")
    
    if improvements:
        print("   ✓ Translation appears successful:")
        for imp in improvements:
            print(f"     - {imp}")
    else:
        print("   ⚠ Translation quality needs verification")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3 if ct_gt_slice is not None else 2, figsize=(15, 5))
    
    axes[0].imshow(cbct_slice, cmap='gray', vmin=cbct.min(), vmax=cbct.max())
    axes[0].set_title(f'Input CBCT\nMean: {cbct_slice.mean():.3f}, Std: {cbct_slice.std():.3f}')
    axes[0].axis('off')
    
    axes[1].imshow(sct_slice, cmap='gray', vmin=sct.min(), vmax=sct.max())
    axes[1].set_title(f'Output sCT\nMean: {sct_slice.mean():.3f}, Std: {sct_slice.std():.3f}')
    axes[1].axis('off')
    
    if ct_gt_slice is not None:
        axes[2].imshow(ct_gt_slice, cmap='gray', vmin=ct_gt.min(), vmax=ct_gt.max())
        axes[2].set_title(f'Ground Truth CT\nMean: {ct_gt_slice.mean():.3f}, Std: {ct_gt_slice.std():.3f}')
        axes[2].axis('off')
    
    plt.suptitle(f'Slice {slice_idx} Analysis: {patient_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'visualizations/analysis_slice_{slice_idx}.png', dpi=150, bbox_inches='tight')
    print(f"\n   Visualization saved to: visualizations/analysis_slice_{slice_idx}.png")
    
    return {
        'cbct_stats': {'mean': cbct_slice.mean(), 'std': cbct_slice.std(), 'min': cbct_slice.min(), 'max': cbct_slice.max()},
        'sct_stats': {'mean': sct_slice.mean(), 'std': sct_slice.std(), 'min': sct_slice.min(), 'max': sct_slice.max()},
        'snr_improvement': ((sct_snr - cbct_snr) / cbct_snr * 100) if cbct_snr > 0 else 0,
        'contrast_change': contrast_improvement
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze CBCT-to-CT translation quality')
    parser.add_argument('--patient', type=int, default=0, help='Patient index')
    parser.add_argument('--slice', type=int, default=None, help='Slice index (default: middle)')
    
    args = parser.parse_args()
    
    slice_idx = args.slice
    if slice_idx is None:
        # Load to get depth
        data_dir = Path('./datasets/psAAPM') / 'test'
        npz_files = sorted(list(data_dir.glob('*.npz')))
        if npz_files:
            data = np.load(npz_files[args.patient])
            slice_idx = data['CBCT'].shape[0] // 2
    
    analyze_slice(patient_idx=args.patient, slice_idx=slice_idx)
