#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Convert model_results.json to markdown table format

import json

# Read the JSON file
with open('model_results.json', 'r') as f:
    results = json.load(f)

# Find best values for highlighting
best_notdog = max(results[m]['pct_correct_notdogs'] for m in results.keys())
best_dogs = max(results[m]['pct_correct_dogs'] for m in results.keys())
best_breeds = max(results[m]['pct_correct_breed'] for m in results.keys())
best_match = max(results[m]['pct_match'] for m in results.keys())

# Generate markdown table
markdown = []
markdown.append("# Project Results\n")
markdown.append("## Image Count Summary\n")
markdown.append(f"- **# Total Images:** {results['resnet']['n_images']}")
markdown.append(f"- **# Dog Images:** {results['resnet']['n_dogs_img']}")
markdown.append(f"- **# Not-a-Dog Images:** {results['resnet']['n_notdogs_img']}\n")

markdown.append("## CNN Model Performance Comparison\n")
markdown.append("| CNN Model Architecture | % Not-a-Dog Correct | % Dogs Correct | % Breeds Correct | % Match Labels |")
markdown.append("|------------------------|---------------------|----------------|------------------|----------------|")

# Add each model's row
for model in ['resnet', 'alexnet', 'vgg']:
    stats = results[model]
    model_name = model.capitalize()
    
    notdog_pct = stats['pct_correct_notdogs']
    dogs_pct = stats['pct_correct_dogs']
    breeds_pct = stats['pct_correct_breed']
    match_pct = stats['pct_match']
    
    # Add asterisk for best performance
    notdog_str = f"{notdog_pct:.1f}%{' *' if notdog_pct == best_notdog else ''}"
    dogs_str = f"{dogs_pct:.1f}%{' *' if dogs_pct == best_dogs else ''}"
    breeds_str = f"{breeds_pct:.1f}%{' *' if breeds_pct == best_breeds else ''}"
    match_str = f"{match_pct:.1f}%{' *' if match_pct == best_match else ''}"
    
    markdown.append(f"| {model_name} | {notdog_str} | {dogs_str} | {breeds_str} | {match_str} |")

markdown.append("\n* indicates best performance for that metric\n")

# Print to console
print("\n".join(markdown))

# Save to file
with open('results_table.md', 'w', encoding='utf-8') as f:
    f.write("\n".join(markdown))

print("\n✓ Markdown table saved to 'results_table.md'")

