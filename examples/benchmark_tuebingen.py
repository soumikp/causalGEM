#!/usr/bin/env python
"""
Benchmark causalGEM on Tuebingen Cause-Effect Pairs Dataset.

This script demonstrates how to use the causalGEM package on the standard
Tuebingen benchmark dataset for pairwise causal discovery.

Requirements:
    pip install cdt  # For loading benchmark data (optional)
"""

import numpy as np
from sklearn.metrics import accuracy_score

# Check if cdt is available
try:
    from cdt.data import load_dataset
    HAS_CDT = True
except ImportError:
    HAS_CDT = False
    print("Note: cdt package not installed. Install with: pip install cdt")
    print("Using synthetic benchmark data instead.\n")

from causalgem import estimate_causal_direction
from causalgem.simulation import generate_benchmark_pair


def load_tuebingen_data():
    """Load Tuebingen cause-effect pairs dataset."""
    if HAS_CDT:
        data, labels = load_dataset('tuebingen')
        return data, labels
    else:
        # Generate synthetic benchmark
        print("Generating synthetic benchmark pairs...")
        n_pairs = 20
        data = []
        labels = []
        
        pair_types = ['anm_square', 'anm_cube', 'anm_exp', 'linear']
        for i in range(n_pairs):
            ptype = pair_types[i % len(pair_types)]
            x, y, direction = generate_benchmark_pair(ptype, n=500, seed=i)
            data.append({'A': x, 'B': y})
            labels.append(direction if direction != 0 else 1)
        
        return data, labels


def run_benchmark():
    """Run benchmark evaluation using causalGEM."""
    print("=" * 60)
    print("causalGEM Benchmark: Generative Exposure Model")
    print("=" * 60)
    
    # Load data
    data, labels = load_tuebingen_data()
    n_pairs = len(data) if isinstance(data, list) else len(data)
    
    print(f"\nNumber of pairs: {n_pairs}")
    print("-" * 60)
    
    predictions = []
    results_list = []
    
    for i in range(n_pairs):
        # Get pair data
        if isinstance(data, list):
            pair = data[i]
            x, y = pair['A'], pair['B']
        else:
            pair = data.iloc[i]
            x, y = pair['A'].values, pair['B'].values
        
        # Run GEM estimation
        try:
            result = estimate_causal_direction(x, y)
            pred = result.decision
            results_list.append(result)
        except Exception as e:
            print(f"Pair {i+1}: Error - {e}")
            pred = 0
            results_list.append(None)
        
        predictions.append(pred)
        
        # Get true label
        if isinstance(labels, list):
            true_label = labels[i]
        else:
            true_label = int(labels.iloc[i])
        
        # Print result
        direction_map = {1: "X→Y", -1: "Y→X", 0: "???"}
        correct = "✓" if pred == true_label else "✗"
        
        if results_list[-1] is not None:
            ci_str = f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
            print(f"Pair {i+1:3d}: C={result.delta:+.3f} {ci_str}  "
                  f"Pred={direction_map[pred]:4s}  True={direction_map.get(true_label, '?'):4s}  {correct}")
        else:
            print(f"Pair {i+1:3d}: Error")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    predictions = np.array(predictions)
    if isinstance(labels, list):
        true_labels = np.array(labels)
    else:
        true_labels = labels.values.astype(int)
    
    # Count predictions
    n_xy = np.sum(predictions == 1)
    n_yx = np.sum(predictions == -1)
    n_unk = np.sum(predictions == 0)
    
    print(f"\nPrediction counts:")
    print(f"  X → Y: {n_xy}")
    print(f"  Y → X: {n_yx}")
    print(f"  Inconclusive: {n_unk}")
    
    # Accuracy on decisive predictions
    decisive_mask = predictions != 0
    if np.sum(decisive_mask) > 0:
        decisive_acc = accuracy_score(
            true_labels[decisive_mask], 
            predictions[decisive_mask]
        )
        print(f"\nAccuracy (decisive only): {decisive_acc:.1%}")
        print(f"Coverage (decisive/total): {np.mean(decisive_mask):.1%}")
    
    # Overall accuracy (treating inconclusive as wrong)
    overall_acc = accuracy_score(true_labels, predictions)
    print(f"Overall accuracy: {overall_acc:.1%}")
    
    return predictions, true_labels, results_list


def compare_with_baselines():
    """Compare causalGEM with other methods if available."""
    if not HAS_CDT:
        print("\nNote: Install 'cdt' package for comparison with ANM and CDS methods.")
        return
    
    print("\n" + "=" * 60)
    print("Comparison with Baseline Methods")
    print("=" * 60)
    
    from cdt.causality.pairwise import ANM, CDS
    
    data, labels = load_dataset('tuebingen')
    n_pairs = len(data)
    
    # Define methods
    def gem_predict(pair):
        result = estimate_causal_direction(pair['A'].values, pair['B'].values)
        return result.decision
    
    def anm_predict(pair):
        try:
            score = ANM().predict_proba(pair)
            return 1 if score > 0 else (-1 if score < 0 else 0)
        except:
            return 0
    
    def cds_predict(pair):
        try:
            score = CDS().predict_proba(pair)
            return 1 if score > 0 else (-1 if score < 0 else 0)
        except:
            return 0
    
    methods = {
        'causalGEM': gem_predict,
        'ANM': anm_predict,
        'CDS': cds_predict,
    }
    
    results = {name: [] for name in methods}
    
    print("\nRunning methods...")
    for i in range(n_pairs):
        pair = data.iloc[i]
        for name, method in methods.items():
            try:
                pred = method(pair)
            except:
                pred = 0
            results[name].append(pred)
    
    # Print comparison
    print("\nMethod Comparison:")
    print("-" * 50)
    print(f"{'Method':<15} {'Accuracy':>10} {'Coverage':>10}")
    print("-" * 50)
    
    true_labels = labels.values.astype(int)
    
    for name, preds in results.items():
        preds = np.array(preds)
        decisive = preds != 0
        
        if np.sum(decisive) > 0:
            acc = accuracy_score(true_labels[decisive], preds[decisive])
            cov = np.mean(decisive)
            print(f"{name:<15} {acc:>9.1%} {cov:>9.1%}")
        else:
            print(f"{name:<15} {'N/A':>10} {'0.0%':>10}")


if __name__ == "__main__":
    # Run main benchmark
    predictions, true_labels, results = run_benchmark()
    
    # Compare with baselines
    compare_with_baselines()
    
    print("\n" + "=" * 60)
    print("Done!")
