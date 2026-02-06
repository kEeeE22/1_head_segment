"""
Comprehensive import test script for the cleaned codebase.
Run this to verify all modules are correctly set up.
"""

def test_dependencies():
    """Test core dependencies."""
    print("=" * 60)
    print("Testing Core Dependencies")
    print("=" * 60)
    
    try:
        import torch
        print(f"[OK] torch {torch.__version__}")
    except ImportError as e:
        print(f"[FAIL] torch: {e}")
        return False
    
    try:
        import cv2
        print(f"[OK] opencv {cv2.__version__}")
    except ImportError as e:
        print(f"[FAIL] opencv: {e}")
        return False
    
    try:
        import numpy as np
        print(f"[OK] numpy {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] numpy: {e}")
        return False
    
    try:
        import segmentation_models_pytorch as smp
        print(f"[OK] segmentation_models_pytorch {smp.__version__}")
    except ImportError as e:
        print(f"[FAIL] segmentation_models_pytorch: {e}")
        return False
    
    try:
        import albumentations
        print(f"[OK] albumentations {albumentations.__version__}")
    except ImportError as e:
        print(f"[FAIL] albumentations: {e}")
        return False
    
    return True


def test_config():
    """Test config module."""
    print("\n" + "=" * 60)
    print("Testing Config Module")
    print("=" * 60)
    
    try:
        from config import (
            LOG_DIR, 
            DEFAULT_DATA_FOLDER, 
            DPVR_NUM_ROOM_CLASSES,
            DPVR_NUM_BOUNDARY_CLASSES,
            NUM_CLASSES,
            DEVICE
        )
        print(f"[OK] config imports OK")
        print(f"  - LOG_DIR: {LOG_DIR}")
        print(f"  - DEVICE: {DEVICE}")
        print(f"  - DPVR_NUM_ROOM_CLASSES: {DPVR_NUM_ROOM_CLASSES}")
        print(f"  - NUM_CLASSES (wall model): {NUM_CLASSES}")
        return True
    except ImportError as e:
        print(f"[FAIL] config: {e}")
        return False


def test_net_module():
    """Test net module."""
    print("\n" + "=" * 60)
    print("Testing Net Module")
    print("=" * 60)
    
    try:
        from net import DFPmodel, create_dfpmodel
        print("[OK] DPVR model imports OK")
    except ImportError as e:
        print(f"[FAIL] DPVR model: {e}")
        return False
    
    try:
        from net import create_model, get_criterion, get_optimizer
        print("[OK] Wall model imports OK")
    except ImportError as e:
        print(f"[FAIL] Wall model: {e}")
        return False
    
    return True


def test_data_module():
    """Test data_class module."""
    print("\n" + "=" * 60)
    print("Testing Data Module")
    print("=" * 60)
    
    try:
        from data_class import DPVR_cubicasa
        print("[OK] DPVR_cubicasa dataset OK")
    except ImportError as e:
        print(f"[FAIL] DPVR_cubicasa: {e}")
        return False
    
    try:
        from data_class import FloorplanSVG, get_train_transform, get_val_transform
        print("[OK] FloorplanSVG dataset OK")
        print("[OK] Transform functions OK")
    except ImportError as e:
        print(f"[FAIL] FloorplanSVG: {e}")
        return False
    
    return True


def test_utils_module():
    """Test utils module."""
    print("\n" + "=" * 60)
    print("Testing Utils Module")
    print("=" * 60)
    
    try:
        from utils.losses import balanced_entropy, cross_two_tasks_weight, get_dpvr_criterion
        print("[OK] Loss functions OK")
    except ImportError as e:
        print(f"[FAIL] Loss functions: {e}")
        return False
    
    try:
        from utils import train_full_metrics
        print("[OK] Training utilities OK")
    except ImportError as e:
        print(f"[FAIL] Training utilities: {e}")
        return False
    
    return True


def test_model_instantiation():
    """Test model instantiation."""
    print("\n" + "=" * 60)
    print("Testing Model Instantiation")
    print("=" * 60)
    
    try:
        from net import DFPmodel, create_model
        import torch
        
        # Test DPVR model
        dpvr_model = DFPmodel(pretrained=False, freeze=False)
        print("[OK] DPVR model instantiation OK")
        
        # Test Wall model
        wall_model = create_model(encoder_name='resnet34', encoder_weights=None)
        print("[OK] Wall model instantiation OK")
        
        # Test forward pass
        test_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            logits_r, logits_cw = dpvr_model(test_input)
            print(f"[OK] DPVR forward pass OK - Room: {logits_r.shape}, Boundary: {logits_cw.shape}")
            
            wall_output = wall_model(test_input)
            print(f"[OK] Wall model forward pass OK - Output: {wall_output.shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Model instantiation: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CUBICASA SEGMENTATION MODULE - IMPORT TEST")
    print("=" * 60)
    
    results = []
    
    results.append(("Dependencies", test_dependencies()))
    results.append(("Config", test_config()))
    results.append(("Net Module", test_net_module()))
    results.append(("Data Module", test_data_module()))
    results.append(("Utils Module", test_utils_module()))
    results.append(("Model Instantiation", test_model_instantiation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "[OK]" if result else "[FAIL]"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[ERROR] SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
