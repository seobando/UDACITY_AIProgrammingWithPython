#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Script to run models and generate table - saves progress incrementally

import sys
import json
import os
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats

RESULTS_FILE = 'model_results.json'

def run_model_and_save(image_dir, model, dogfile):
    """Run a model and save results to file"""
    print(f"\n{'='*60}")
    print(f"Running {model.upper()} model...")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Run classification pipeline
    results = get_pet_labels(image_dir)
    classify_images(image_dir, results, model)
    adjust_results4_isadog(results, dogfile)
    stats = calculates_results_stats(results)
    
    # Load existing results or create new dict
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    # Save this model's results
    all_results[model] = {
        'pct_correct_notdogs': stats['pct_correct_notdogs'],
        'pct_correct_dogs': stats['pct_correct_dogs'],
        'pct_correct_breed': stats['pct_correct_breed'],
        'pct_match': stats['pct_match'],
        'n_images': stats['n_images'],
        'n_dogs_img': stats['n_dogs_img'],
        'n_notdogs_img': stats['n_notdogs_img']
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ {model.upper()} completed and saved!")
    print(f"  Not-a-Dog Correct: {stats['pct_correct_notdogs']:.1f}%")
    print(f"  Dogs Correct: {stats['pct_correct_dogs']:.1f}%")
    print(f"  Breeds Correct: {stats['pct_correct_breed']:.1f}%")
    print(f"  Match Labels: {stats['pct_match']:.1f}%")
    sys.stdout.flush()
    
    return stats

def generate_table_from_file():
    """Generate table from saved results file"""
    if not os.path.exists(RESULTS_FILE):
        print("No results file found. Run models first.")
        return
    
    with open(RESULTS_FILE, 'r') as f:
        all_results = json.load(f)
    
    models = ['resnet', 'alexnet', 'vgg']
    
    # Check if we have all models
    missing = [m for m in models if m not in all_results]
    if missing:
        print(f"\nStill need to run: {', '.join(m.upper() for m in missing)}")
        print("Current results:")
    
    # Print table
    print("\n" + "="*80)
    print(" " * 25 + "Project Results")
    print("="*80)
    
    if all_results:
        first_model = list(all_results.keys())[0]
        stats = all_results[first_model]
        print("\nImage Count Summary:")
        print("-" * 40)
        print(f"# Total Images: {stats['n_images']}")
        print(f"# Dog Images: {stats['n_dogs_img']}")
        print(f"# Not-a-Dog Images: {stats['n_notdogs_img']}")
    
    print("\n" + "="*80)
    print("CNN Model Performance Comparison")
    print("="*80)
    
    header = f"{'CNN Model Architecture':<25} {'% Not-a-Dog Correct':<20} {'% Dogs Correct':<18} {'% Breeds Correct':<20} {'% Match Labels':<18}"
    print(header)
    print("-" * 80)
    
    # Find best values
    if len(all_results) > 1:
        best_notdog = max(all_results[m]['pct_correct_notdogs'] for m in all_results.keys())
        best_dogs = max(all_results[m]['pct_correct_dogs'] for m in all_results.keys())
        best_breeds = max(all_results[m]['pct_correct_breed'] for m in all_results.keys())
        best_match = max(all_results[m]['pct_match'] for m in all_results.keys())
    else:
        best_notdog = best_dogs = best_breeds = best_match = 0
    
    # Print each model
    for model in models:
        if model in all_results:
            stats = all_results[model]
            model_name = model.capitalize()
            
            notdog_pct = stats['pct_correct_notdogs']
            dogs_pct = stats['pct_correct_dogs']
            breeds_pct = stats['pct_correct_breed']
            match_pct = stats['pct_match']
            
            notdog_str = f"{notdog_pct:.1f}%{' *' if len(all_results) > 1 and notdog_pct == best_notdog else ''}"
            dogs_str = f"{dogs_pct:.1f}%{' *' if len(all_results) > 1 and dogs_pct == best_dogs else ''}"
            breeds_str = f"{breeds_pct:.1f}%{' *' if len(all_results) > 1 and breeds_pct == best_breeds else ''}"
            match_str = f"{match_pct:.1f}%{' *' if len(all_results) > 1 and match_pct == best_match else ''}"
            
            row = f"{model_name:<25} {notdog_str:<20} {dogs_str:<18} {breeds_str:<20} {match_str:<18}"
            print(row)
        else:
            print(f"{model.capitalize():<25} {'(not run yet)':<20}")
    
    print("="*80)
    if len(all_results) > 1:
        print("* indicates best performance for that metric")
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--table-only':
        # Just show table from existing results
        generate_table_from_file()
    else:
        # Run all models
        image_dir = 'pet_images/'
        dogfile = 'dognames.txt'
        
        models = ['resnet', 'alexnet', 'vgg']
        
        print("Starting model classification...")
        print("Results will be saved incrementally.\n")
        
        for model in models:
            try:
                run_model_and_save(image_dir, model, dogfile)
            except Exception as e:
                print(f"\nError with {model.upper()}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("All models completed! Generating final table...")
        print("="*60)
        
        generate_table_from_file()

