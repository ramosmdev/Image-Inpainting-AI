# Training Guide & Hardware Optimization

Training deep neural networks can be incredibly slow if not optimized correctly. Because this project was designed specifically targeting consumer-grade Laptop GPUs (like the NVIDIA RTX 5070), we've implemented multiple PyTorch hardware optimizations out-of-the-box.

You can find all of these in `train.py`.

## GPU Optimizations Implemented

We achieved massive speedups by integrating the following techniques:

1. **CuDNN Benchmark (`torch.backends.cudnn.benchmark = True`)**: 
   Since our image input sizes are fixed (e.g. 256×256), enabling this flag tells PyTorch's backend to automatically profile and choose the most optimized convolution algorithms for your specific GPU architecture before the first batch.
   
2. **Pinned Memory (`pin_memory=True`)**: 
   When the CPU loads batches from the Dataset, applying `pin_memory=True` stores the tensors in page-locked (pinned) memory. This allows the PCIe bus to transfer data to the GPU much faster and asynchronously.

3. **Background Data Loading (`num_workers=4`)**: 
   The disk bottleneck is bypassed by using multiple background CPU processes to read images from the SSD while the GPU is busy training the current batch.

4. **Asynchronous GPU Transfers (`non_blocking=True`)**: 
   When moving our images, masks, and ground-truths to the GPU via `.to(device, non_blocking=True)`, we allow the CPU to continue executing Python code without waiting for the GPU memory allocation to complete.

5. **Console IO Bottleneck Avoidance**: 
   Printing to the terminal is highly asynchronous and blocks the main execution thread. The training loop carefully logs metrics only once every 20 batches, avoiding severe performance degradation.

## Modifying Hyperparameters

All configurations can be found inside `src/config.py`. 

- **Batch Size (`BATCH_SIZE`)**: If you run out of VRAM (OOM error), reduce this number (e.g., from 16 to 8 or 4). If your GPU utilization is low and you have free VRAM, increase it.
- **Learning Rate (`LR`)**: Set to `1e-4` (Adam optimizer). If the loss plateaus too early, you can implement a Learning Rate Scheduler.
- **Image Size (`IMG_SIZE`)**: Currently set to **256×256**. Increasing to 512 will vastly increase generation fidelity but requires exponentially more VRAM and compute time. Going from 128→256 requires halving the batch size (16→8) to compensate for ~4× higher VRAM usage per image.
