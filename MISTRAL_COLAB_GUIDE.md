# Running Mistral 7B on Google Colab

This guide explains how to use the `04_mistral_7b_colab.ipynb` notebook to run the Mistral 7B model on Google Colab with GPU acceleration.

## Steps to Run

1. Upload the `04_mistral_7b_colab.ipynb` notebook to Google Colab
2. Make sure to select a GPU runtime:
   - Go to Runtime -> Change runtime type -> Hardware accelerator -> GPU
3. Run the cells in order:
   - First cell installs the required dependencies
   - Second cell checks GPU availability
   - Third cell loads the Mistral 7B model (this may take a few minutes)
   - Fourth cell defines the domain suggestion function
   - Fifth cell tests the domain suggestion system
   - Sixth cell shows the optimized batch version
   - Seventh cell tests the batch version
   - Eighth cell prepares for fine-tuning (optional)
   - Ninth cell provides conclusion

## Notes

1. The model loading step (third cell) may take 5-10 minutes as it needs to download the model weights
2. The first generation might be slow due to model compilation, subsequent generations will be faster
3. If you encounter out-of-memory errors, try reducing the `max_new_tokens` parameter in the generation function
4. For fine-tuning, you'll need to prepare your dataset in the appropriate format

## Expected Results

The notebook will generate domain name suggestions for sample business descriptions using the Mistral 7B model. Each suggestion will include a domain name and a confidence score.

## Troubleshooting

1. If you get a "CUDA out of memory" error:
   - Reduce the `max_new_tokens` parameter
   - Use a smaller model variant if available
   - Restart the runtime and try again

2. If the model loading fails:
   - Check your internet connection
   - Make sure you have enough disk space (Mistral 7B requires ~15GB)
   - Restart the runtime and try again

3. If generation is slow:
   - The first generation is always slower due to compilation
   - Subsequent generations should be faster