#!/usr/bin/env python3
# Quick script to check progress and show current results

import json
import os

RESULTS_FILE = 'model_results.json'

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    print("\nCurrent Progress:")
    print("="*60)
    models = ['resnet', 'alexnet', 'vgg']
    for model in models:
        if model in results:
            stats = results[model]
            print(f"\n{model.upper()}: ✓ Complete")
            print(f"  Not-a-Dog: {stats['pct_correct_notdogs']:.1f}%")
            print(f"  Dogs: {stats['pct_correct_dogs']:.1f}%")
            print(f"  Breeds: {stats['pct_correct_breed']:.1f}%")
            print(f"  Match: {stats['pct_match']:.1f}%")
        else:
            print(f"\n{model.upper()}: ⏳ Not started or in progress...")
    
    # Show partial table if we have at least one result
    if results:
        print("\n" + "="*80)
        print("Partial Results Table:")
        print("="*80)
        header = f"{'Model':<15} {'% Not-a-Dog':<15} {'% Dogs':<15} {'% Breeds':<15} {'% Match':<15}"
        print(header)
        print("-" * 80)
        for model in models:
            if model in results:
                stats = results[model]
                print(f"{model.capitalize():<15} {stats['pct_correct_notdogs']:>6.1f}%      {stats['pct_correct_dogs']:>6.1f}%      {stats['pct_correct_breed']:>6.1f}%      {stats['pct_match']:>6.1f}%")
else:
    print("No results file found yet. Models are still running...")
    print("This may take 10-15 minutes total.")

