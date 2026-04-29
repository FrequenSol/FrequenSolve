#!/usr/bin/env python3
"""
Test script to demonstrate finding existing files with the same data.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frequensolve.util.data_file import DataArrayFile, save_data_if_new


def main():
    """Test the existing file detection functionality."""

    # Create a test directory
    test_dir = Path("test_existing_file_detection")
    test_dir.mkdir(exist_ok=True)

    print("=== Testing Existing File Detection ===\n")

    # Create some test data
    data1 = np.random.rand(20, 20)
    data2 = np.random.rand(20, 20)  # Different data

    print("1. Saving data1 to first location...")
    path1 = save_data_if_new(data1, test_dir / "data1.npy")
    print(f"Data saved to: {path1}")

    print("\n2. Trying to save data1 to a different location...")
    path2 = save_data_if_new(data1, test_dir / "data1_different_name.npy")
    print(f"Data saved to: {path2}")
    print(f"Note: Should return the original path: {path1}")
    print(f"Are they the same? {path1 == path2}")

    print("\n3. Saving data2 to a new location...")
    path3 = save_data_if_new(data2, test_dir / "data2.npy")
    print(f"Data saved to: {path3}")

    print("\n4. Trying to save data1 again to yet another location...")
    path4 = save_data_if_new(data1, test_dir / "data1_another_name.npy")
    print(f"Data saved to: {path4}")
    print(f"Should return the original path: {path1}")
    print(f"Are they the same? {path1 == path4}")

    print("\n5. Testing with DataArrayFile directly...")
    data_file = DataArrayFile(test_dir / "new_test.npy")

    # Check if data1 exists in any file
    existing_file = data_file.find_existing_file(data1)
    print(f"Existing file for data1: {existing_file}")

    # Check if data2 exists in any file
    existing_file = data_file.find_existing_file(data2)
    print(f"Existing file for data2: {existing_file}")

    print("\n6. Manifest contents:")
    manifest_file = test_dir / "data_manifest.json"
    if manifest_file.exists():
        import json

        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        for filename, info in manifest.items():
            print(f"  {filename}: hash={info['hash'][:16]}...")

    print("\n=== Test Complete ===")
    print(f"Test files created in: {test_dir.absolute()}")


if __name__ == "__main__":
    main()
