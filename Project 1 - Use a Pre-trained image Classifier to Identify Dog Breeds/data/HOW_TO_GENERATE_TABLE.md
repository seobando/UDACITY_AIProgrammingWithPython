# How to Generate the Results Comparison Table

There are two methods to generate the comparison table showing results for all three CNN models (ResNet, AlexNet, VGG):

## Method 1: Automated Script (Recommended)

Use the `generate_results_table.py` script to automatically run all models and generate a formatted comparison table:

```bash
python generate_results_table.py --dir pet_images/ --dogfile dognames.txt
```

This script will:
1. Run classification for all three models (ResNet, AlexNet, VGG)
2. Collect statistics from each model
3. Generate a formatted comparison table showing:
   - % Not-a-Dog Correct
   - % Dogs Correct
   - % Breeds Correct
   - % Match Labels

The script will automatically highlight the best performance for each metric with an asterisk (*).

## Method 2: Manual Batch Processing

### Step 1: Run all three models

**On Windows (PowerShell):**
```powershell
.\run_models_batch.ps1
```

**On Linux/Mac (Bash):**
```bash
sh run_models_batch.sh
```

Or run each model individually:
```bash
python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt > resnet_pet-images.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > alexnet_pet-images.txt
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt > vgg_pet-images.txt
```

### Step 2: Extract statistics from output files

Open each output file and look for the "Results Summary" section. Extract the following percentages:
- `pct_correct_notdogs` → % Not-a-Dog Correct
- `pct_correct_dogs` → % Dogs Correct
- `pct_correct_breed` → % Breeds Correct
- `pct_match` → % Match Labels

### Step 3: Create the table

Create a table with the following structure:

| CNN Model Architecture | % Not-a-Dog Correct | % Dogs Correct | % Breeds Correct | % Match Labels |
|------------------------|---------------------|----------------|------------------|----------------|
| ResNet                 | [value]             | [value]        | [value]          | [value]        |
| AlexNet                | [value]             | [value]        | [value]          | [value]        |
| VGG                    | [value]             | [value]        | [value]          | [value]        |

## Expected Results

Based on typical performance:
- **VGG** usually has the best breed classification accuracy (~93.3%)
- **AlexNet** and **VGG** typically achieve 100% for dog/non-dog classification
- **VGG** usually has the highest match rate (~87.5%)

## Notes

- The automated script (Method 1) is faster and less error-prone
- Running all three models takes approximately 10-15 minutes total
- Make sure you're in the `data` directory when running the scripts
- The comparison table helps identify which model performs best for different metrics

