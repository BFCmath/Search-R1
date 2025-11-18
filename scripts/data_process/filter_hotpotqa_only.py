#!/usr/bin/env python3
"""
Filter the nq_hotpotqa_train dataset to keep only HotpotQA samples
and optionally sample a subset for faster training/validation
"""

import pandas as pd
import os
import argparse

def filter_hotpotqa_data(input_dir: str, output_dir: str, sample_ratio: float = 1.0, random_seed: int = 42):
    """
    Filter dataset to keep only HotpotQA samples and optionally sample a subset
    
    Args:
        input_dir: Input directory with combined dataset
        output_dir: Output directory for filtered dataset
        sample_ratio: Fraction of data to keep (0.0 to 1.0). Default 1.0 (all data)
        random_seed: Random seed for reproducible sampling
    """

    os.makedirs(output_dir, exist_ok=True)

    # Filter training data
    print("="*80)
    print("Filtering training data...")
    print("="*80)
    train_df = pd.read_parquet(os.path.join(input_dir, 'train.parquet'))
    hotpotqa_train = train_df[train_df['data_source'] == 'hotpotqa']
    print(f"Original train samples: {len(train_df)}")
    print(f"HotpotQA train samples: {len(hotpotqa_train)}")
    
    # Sample training data if requested
    if sample_ratio < 1.0:
        original_count = len(hotpotqa_train)
        hotpotqa_train = hotpotqa_train.sample(frac=sample_ratio, random_state=random_seed)
        print(f"📊 Sampling {sample_ratio*100:.1f}% of training data...")
        print(f"   Before sampling: {original_count}")
        print(f"   After sampling:  {len(hotpotqa_train)}")
        print(f"   (Random seed: {random_seed} for reproducibility)")

    # Filter test data
    print("\n" + "="*80)
    print("Filtering test data...")
    print("="*80)
    test_df = pd.read_parquet(os.path.join(input_dir, 'test.parquet'))
    hotpotqa_test = test_df[test_df['data_source'] == 'hotpotqa']
    print(f"Original test samples: {len(test_df)}")
    print(f"HotpotQA test samples: {len(hotpotqa_test)}")
    
    # Sample test data if requested
    if sample_ratio < 1.0:
        original_count = len(hotpotqa_test)
        hotpotqa_test = hotpotqa_test.sample(frac=sample_ratio, random_state=random_seed)
        print(f"📊 Sampling {sample_ratio*100:.1f}% of test data...")
        print(f"   Before sampling: {original_count}")
        print(f"   After sampling:  {len(hotpotqa_test)}")
        print(f"   (Random seed: {random_seed} for reproducibility)")

    # Save filtered datasets
    train_output = os.path.join(output_dir, 'train.parquet')
    test_output = os.path.join(output_dir, 'test.parquet')

    hotpotqa_train.to_parquet(train_output)
    hotpotqa_test.to_parquet(test_output)

    print("\n" + "="*80)
    print("✅ Saved HotpotQA-only data")
    print("="*80)
    print(f"Train data: {train_output}")
    print(f"Test data:  {test_output}")
    print(f"Train samples: {len(hotpotqa_train)}")
    print(f"Test samples:  {len(hotpotqa_test)}")
    
    if sample_ratio < 1.0:
        print(f"\n⚡ Dataset reduced to {sample_ratio*100:.1f}% for faster training/validation!")

    return len(hotpotqa_train), len(hotpotqa_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter nq_hotpotqa_train to keep only HotpotQA samples and optionally sample a subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep all HotpotQA data (default)
  python filter_hotpotqa_only.py
  
  # Keep only 10% for faster training
  python filter_hotpotqa_only.py --sample_ratio 0.1
  
  # Keep 50% with custom directories
  python filter_hotpotqa_only.py --input_dir ./data/nq_hotpotqa_train --output_dir ./data/hotpotqa_50pct --sample_ratio 0.5
        """
    )
    parser.add_argument('--input_dir', default='./data/nq_hotpotqa_train',
                       help='Input directory containing the combined dataset')
    parser.add_argument('--output_dir', default='./data/hotpotqa_only',
                       help='Output directory for HotpotQA-only dataset')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Fraction of data to keep (0.0 to 1.0). Default: 1.0 (all data). Use 0.1 for 10%%')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible sampling. Default: 42')

    args = parser.parse_args()
    
    # Validate sample_ratio
    if not 0.0 < args.sample_ratio <= 1.0:
        parser.error("sample_ratio must be between 0.0 and 1.0")

    print("\n" + "="*80)
    print("🔍 HotpotQA Dataset Filter & Sampler")
    print("="*80)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Sample: {args.sample_ratio*100:.1f}% of data")
    if args.sample_ratio < 1.0:
        print(f"Seed:   {args.random_seed}")
    print("="*80 + "\n")
    
    train_count, test_count = filter_hotpotqa_data(
        args.input_dir, 
        args.output_dir,
        args.sample_ratio,
        args.random_seed
    )
    
    print("\n" + "="*80)
    print(f"✅ Successfully created dataset with {train_count} train and {test_count} test samples!")
    print("="*80)
