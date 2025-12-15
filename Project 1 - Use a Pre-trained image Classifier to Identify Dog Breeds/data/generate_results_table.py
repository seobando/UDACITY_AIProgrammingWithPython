#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/generate_results_table.py
#                                                                             
# PURPOSE: Runs all three CNN models (ResNet, AlexNet, VGG) and generates a 
#          comparison table showing their performance metrics.
#
# Usage: python generate_results_table.py --dir pet_images/ --dogfile dognames.txt

import sys
import argparse
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats

def run_model_classification(image_dir, model, dogfile):
    """
    Runs the classification pipeline for a specific model and returns statistics.
    
    Parameters:
      image_dir - Path to folder of images (string)
      model - CNN model architecture: 'resnet', 'alexnet', or 'vgg' (string)
      dogfile - Path to text file with dog names (string)
    
    Returns:
      results_stats_dic - Dictionary with statistics for the model
    """
    # Get pet labels from filenames
    results = get_pet_labels(image_dir)
    
    # Classify images using the specified model
    classify_images(image_dir, results, model)
    
    # Adjust results to determine if labels are dogs
    adjust_results4_isadog(results, dogfile)
    
    # Calculate statistics
    results_stats = calculates_results_stats(results)
    
    return results_stats

def generate_comparison_table(results_dict):
    """
    Generates a formatted comparison table from results for all three models.
    
    Parameters:
      results_dict - Dictionary with model names as keys and stats dictionaries as values
    """
    models = ['resnet', 'alexnet', 'vgg']
    
    # Print header
    print("\n" + "="*80)
    print(" " * 25 + "Project Results")
    print("="*80)
    
    # Print image count summary
    print("\nImage Count Summary:")
    print("-" * 40)
    # Get counts from first model (all should be the same)
    first_model = models[0]
    stats = results_dict[first_model]
    print(f"# Total Images: {stats['n_images']}")
    print(f"# Dog Images: {stats['n_dogs_img']}")
    print(f"# Not-a-Dog Images: {stats['n_notdogs_img']}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("CNN Model Performance Comparison")
    print("="*80)
    
    # Table header
    header = f"{'CNN Model Architecture':<25} {'% Not-a-Dog Correct':<20} {'% Dogs Correct':<18} {'% Breeds Correct':<20} {'% Match Labels':<18}"
    print(header)
    print("-" * 80)
    
    # Find best values for highlighting
    best_notdog = max(results_dict[m]['pct_correct_notdogs'] for m in models)
    best_dogs = max(results_dict[m]['pct_correct_dogs'] for m in models)
    best_breeds = max(results_dict[m]['pct_correct_breed'] for m in models)
    best_match = max(results_dict[m]['pct_match'] for m in models)
    
    # Print each model's results
    for model in models:
        stats = results_dict[model]
        model_name = model.capitalize()
        
        # Format percentages
        notdog_pct = stats['pct_correct_notdogs']
        dogs_pct = stats['pct_correct_dogs']
        breeds_pct = stats['pct_correct_breed']
        match_pct = stats['pct_match']
        
        # Mark best values with asterisk
        notdog_str = f"{notdog_pct:.1f}%{' *' if notdog_pct == best_notdog else ''}"
        dogs_str = f"{dogs_pct:.1f}%{' *' if dogs_pct == best_dogs else ''}"
        breeds_str = f"{breeds_pct:.1f}%{' *' if breeds_pct == best_breeds else ''}"
        match_str = f"{match_pct:.1f}%{' *' if match_pct == best_match else ''}"
        
        row = f"{model_name:<25} {notdog_str:<20} {dogs_str:<18} {breeds_str:<20} {match_str:<18}"
        print(row)
    
    print("="*80)
    print("* indicates best performance for that metric")
    print("="*80 + "\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/', 
                        help='path to folder of images')
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='path to text file with dog names')
    args = parser.parse_args()
    
    print("Running classification for all three models...")
    print("This may take a few minutes...\n")
    sys.stdout.flush()
    
    # Run classification for each model
    results_dict = {}
    models = ['resnet', 'alexnet', 'vgg']
    
    for model in models:
        print(f"Processing {model.upper()} model...")
        sys.stdout.flush()
        try:
            stats = run_model_classification(args.dir, model, args.dogfile)
            results_dict[model] = stats
            print(f"  ✓ {model.upper()} completed\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"  ✗ Error processing {model.upper()}: {e}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Generate and print comparison table
    generate_comparison_table(results_dict)

if __name__ == "__main__":
    main()

