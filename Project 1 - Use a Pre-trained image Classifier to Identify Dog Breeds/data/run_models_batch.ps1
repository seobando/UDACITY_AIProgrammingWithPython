# PowerShell script to run all three models and save output to text files
# Usage: .\run_models_batch.ps1

Write-Host "Running all three CNN models..."
Write-Host "This may take several minutes..."
Write-Host ""

Write-Host "Running ResNet..."
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt > resnet_pet-images.txt
Write-Host "ResNet completed. Results saved to resnet_pet-images.txt"

Write-Host ""
Write-Host "Running AlexNet..."
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > alexnet_pet-images.txt
Write-Host "AlexNet completed. Results saved to alexnet_pet-images.txt"

Write-Host ""
Write-Host "Running VGG..."
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt > vgg_pet-images.txt
Write-Host "VGG completed. Results saved to vgg_pet-images.txt"

Write-Host ""
Write-Host "All models completed!"
Write-Host "Check the output files for detailed results."
Write-Host "Use generate_results_table.py to create a comparison table."

