# Project Results

## Image Count Summary
- **# Total Images:** 40
- **# Dog Images:** 30
- **# Not-a-Dog Images:** 10

## CNN Model Performance Comparison

| CNN Model Architecture | % Not-a-Dog Correct | % Dogs Correct | % Breeds Correct | % Match Labels |
|------------------------|---------------------|----------------|------------------|----------------|
| ResNet | 90.0% | 100.0% * | 90.0% | 82.5% |
| AlexNet | 100.0% * | 100.0% * | 80.0% | 75.0% |
| VGG | 100.0% * | 100.0% * | 93.3% * | 87.5% * |

* indicates best performance for that metric

## Summary

- **Best Not-a-Dog Classification:** AlexNet & VGG (100.0%)
- **Best Dog Classification:** All models (100.0%)
- **Best Breed Classification:** VGG (93.3%)
- **Best Match Rate:** VGG (87.5%)

**Overall Winner:** VGG performs best overall, especially for breed classification and match accuracy.
