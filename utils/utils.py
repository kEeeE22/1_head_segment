"""
Utility functions for dataset preparation and validation.
"""

import torch
from tqdm import tqdm
import numpy as np

def patch_house_file(file_path):
    """
    Patch the house.py file to fix overflow and division errors.
    
    This function fixes two issues:
    1. Overflow error: Changes dtype from uint8 to int32
    2. Division by zero: Adds check before division
    
    Args:
        file_path (str): Path to the house.py file to patch
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for line in lines:
            # 1. Fix Overflow (uint8 -> int32)
            if 'self.wall_ids = np.empty((height, width), dtype=np.uint8)' in line:
                f.write(line.replace('np.uint8', 'np.int32'))
            
            # 2. Fix Divide by Zero (line 952)
            elif 'res = res / float(i)' in line:
                # Check if i > 0 before division, otherwise return 1.0
                f.write('        res = res / float(i) if i > 0 else 1.0\n')
            
            else:
                f.write(line)

    print("✅ Đã sửa xong cả 2 lỗi trong house.py!")
    print("⚠️  Lưu ý: Bạn cần restart Python session để áp dụng thay đổi.")


def verify_dataset_loading(dataset, max_samples=None):
    """
    Verify that dataset can be loaded without errors.
    
    Args:
        dataset: FloorplanSVG dataset instance
        max_samples (int, optional): Maximum number of samples to check.
                                     If None, checks all samples.
    
    Returns:
        dict: Summary with error count and error details
    """
    print("🚀 Bắt đầu kiểm tra Dataset...")
    
    errors = []
    overflow_count = 0
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    # Scan through dataset
    for i in tqdm(range(num_samples), desc="Checking samples"):
        try:
            # Call __getitem__ - this is where overflow errors occur
            sample = dataset[i]
            
            # Check for NaN/inf values from division by zero
            if torch.isnan(sample['wall_mask']).any():
                print(f"\n⚠️ Cảnh báo: Phát hiện NaN tại index {i} ({dataset.folders[i]})")

        except OverflowError as e:
            overflow_count += 1
            folder_name = dataset.folders[i]
            errors.append(f"Index {i} ({folder_name}): Lỗi Overflow vẫn còn! -> {e}")
            
        except Exception as e:
            folder_name = dataset.folders[i]
            errors.append(f"Index {i} ({folder_name}): Lỗi khác -> {type(e).__name__}: {e}")

    # Summary
    print("\n" + "="*50)
    print("KẾT QUẢ KIỂM TRA")
    print("="*50)
    
    if len(errors) == 0:
        print("🎉 Tuyệt vời! Không còn lỗi OverflowError nào.")
        print("✅ Dataset đã hoạt động chính xác.")
    else:
        print(f"❌ Vẫn còn {len(errors)} mẫu bị lỗi.")
        for err in errors[:10]:  # Print first 10 errors
            print(err)
        if len(errors) > 10:
            print(f"... và {len(errors) - 10} lỗi khác.")
            
    print("="*50)
    
    return {
        'total_checked': num_samples,
        'error_count': len(errors),
        'overflow_count': overflow_count,
        'errors': errors
    }
