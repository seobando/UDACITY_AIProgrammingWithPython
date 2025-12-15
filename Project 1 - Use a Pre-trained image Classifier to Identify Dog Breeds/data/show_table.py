#!/usr/bin/env python3
import json

with open('model_results.json', 'r') as f:
    results = json.load(f)

print("\n" + "="*80)
print(" " * 25 + "Project Results")
print("="*80)

# Image count summary
stats = results['resnet']
print("\nImage Count Summary:")
print("-" * 40)
print(f"# Total Images: {stats['n_images']}")
print(f"# Dog Images: {stats['n_dogs_img']}")
print(f"# Not-a-Dog Images: {stats['n_notdogs_img']}")

# Comparison table
print("\n" + "="*80)
print("CNN Model Performance Comparison")
print("="*80)

header = f"{'CNN Model Architecture':<25} {'% Not-a-Dog Correct':<20} {'% Dogs Correct':<18} {'% Breeds Correct':<20} {'% Match Labels':<18}"
print(header)
print("-" * 80)

# Find best values
best_notdog = max(results[m]['pct_correct_notdogs'] for m in results.keys())
best_dogs = max(results[m]['pct_correct_dogs'] for m in results.keys())
best_breeds = max(results[m]['pct_correct_breed'] for m in results.keys())
best_match = max(results[m]['pct_match'] for m in results.keys())

# Print each model
for model in ['resnet', 'alexnet', 'vgg']:
    stats = results[model]
    model_name = model.capitalize()
    
    notdog_pct = stats['pct_correct_notdogs']
    dogs_pct = stats['pct_correct_dogs']
    breeds_pct = stats['pct_correct_breed']
    match_pct = stats['pct_match']
    
    notdog_str = f"{notdog_pct:.1f}%{' *' if notdog_pct == best_notdog else ''}"
    dogs_str = f"{dogs_pct:.1f}%{' *' if dogs_pct == best_dogs else ''}"
    breeds_str = f"{breeds_pct:.1f}%{' *' if breeds_pct == best_breeds else ''}"
    match_str = f"{match_pct:.1f}%{' *' if match_pct == best_match else ''}"
    
    row = f"{model_name:<25} {notdog_str:<20} {dogs_str:<18} {breeds_str:<20} {match_str:<18}"
    print(row)

print("="*80)
print("* indicates best performance for that metric")
print("="*80 + "\n")

