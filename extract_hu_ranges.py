#!/usr/bin/env python3
"""
Extract original HU min/max values from original CT files
"""
import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

def extract_hu_ranges(dataset_dir, output_file='hu_ranges.json'):
    """
    Extract HU ranges from original CT files in the dataset
    
    Args:
        dataset_dir: Path to the dataset directory (e.g., /Volumes/T9/test_psCBCT_AAPM)
        output_file: Output JSON file to save HU ranges
    """
    dataset_path = Path(dataset_dir)
    hu_ranges = {}
    
    # Find all directories that might contain CT files
    for case_dir in sorted(dataset_path.iterdir()):
        if not case_dir.is_dir():
            continue
            
        print(f"Processing: {case_dir.name}")
        
        # Look for CT_plan_50.nrrd files
        ct_files = list(case_dir.glob('*CT_plan_50.nrrd'))
        
        if not ct_files:
            print(f"  No CT_plan_50.nrrd found in {case_dir.name}")
            continue
        
        # Use the first CT file found (there should be one per case)
        ct_file = ct_files[0]
        
        try:
            # Load the original CT
            ct_img = sitk.ReadImage(str(ct_file))
            ct_array = sitk.GetArrayFromImage(ct_img)
            
            # Get HU statistics
            min_hu = float(ct_array.min())
            max_hu = float(ct_array.max())
            mean_hu = float(ct_array.mean())
            std_hu = float(ct_array.std())
            
            # Store with case name as key
            case_name = case_dir.name
            hu_ranges[case_name] = {
                'min_HU': min_hu,
                'max_HU': max_hu,
                'mean_HU': mean_hu,
                'std_HU': std_hu,
                'ct_file': str(ct_file)
            }
            
            print(f"  {case_name}: HU range [{min_hu:.1f}, {max_hu:.1f}], mean: {mean_hu:.1f}")
            
        except Exception as e:
            print(f"  Error processing {ct_file}: {e}")
            continue
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(hu_ranges, f, indent=2)
    
    print(f"\nSaved HU ranges to: {output_file}")
    print(f"Total cases processed: {len(hu_ranges)}")
    
    return hu_ranges

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract HU ranges from original CT files')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='hu_ranges.json',
                       help='Output JSON file name')
    
    args = parser.parse_args()
    extract_hu_ranges(args.dataset_dir, args.output)
