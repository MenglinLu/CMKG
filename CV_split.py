# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os


def create_cv_splits(data_path, output_dir, label_col='label_map', n_folds=5, 
                     train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, 
                     random_state=42):
    """
    Create stratified 5-fold cross-validation splits.
    
    Args:
        data_path: Path to the full dataset CSV
        output_dir: Directory to save split CSV files
        label_col: Column name containing the disease label
        n_folds: Number of folds (default 5)
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.10)
        test_ratio: Proportion for testing (default 0.20)
        random_state: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train + Val + Test ratios must sum to 1.0"
    
    # Load full dataset
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples with {df[label_col].nunique()} classes.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # First split: full data → 5 folds (for test set rotation)
    skf_test = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for fold_idx, (non_test_idx, test_idx) in enumerate(skf_test.split(df, df[label_col])):
        
        non_test_df = df.iloc[non_test_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Second split: non-test (80%) → val (10%) + train (70%)
        # val_ratio / (1 - test_ratio) = 0.10 / 0.80 = 0.125
        val_split_ratio = val_ratio / (1 - test_ratio)
        
        skf_val = StratifiedKFold(n_splits=int(1/val_split_ratio), 
                                   shuffle=True, 
                                   random_state=random_state + fold_idx)
        
        # Get the first split's validation indices from non-test data
        for train_rel_idx, val_rel_idx in skf_val.split(non_test_df, non_test_df[label_col]):
            train_df = non_test_df.iloc[train_rel_idx].copy()
            val_df = non_test_df.iloc[val_rel_idx].copy()
            break  # Only need the first split
        
        # Save CSV files for this fold
        train_path = os.path.join(output_dir, f'train_fold_{fold_idx}.csv')
        val_path = os.path.join(output_dir, f'val_fold_{fold_idx}.csv')
        test_path = os.path.join(output_dir, f'test_fold_{fold_idx}.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)



# ==================== Usage Example ====================

if __name__ == '__main__':
    
    create_cv_splits(
        data_path='data.csv',      # ← 替换为您的数据路径
        output_dir='./cv_splits/',       # ← 替换为输出目录
        label_col='label_map',           # ← 替换为您的标签列名
        random_state=42
    )