import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from efficientvit.models.utils import resize


from efficientvit.seg_model_zoo import create_seg_model

def print_model_size(model):
    torch.cuda.reset_peak_memory_stats()
    model.to('cuda')  # Move model to GPU to measure GPU memory
    torch.cuda.synchronize()  # Wait for move to complete
    print(f'Model memory footprint: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')




def measure_inference_time(model, input_tensor, num_warmup=5, num_runs=50):
    """
    Measures the inference time of a PyTorch model using CUDA events.

    Parameters:
    - model: the PyTorch model to evaluate.
    - input_tensor: the input tensor to feed to the model.
    - num_warmup: number of warm-up runs before measurements.
    - num_runs: number of timed runs for averaging inference time.
    """
    # Ensure model is in evaluation mode and moved to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    input_tensor = input_tensor.to(device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Timing inference
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    durations = []
    memory_stats = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.reset_peak_memory_stats()
            
            start_event.record()
            outputs = model(input_tensor) 
            outputs = resize(outputs, size=[1200,1920])
            _, pred = torch.max(outputs, 1)
            pred = pred.byte().cpu()
            end_event.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            # memory usage
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert bytes to megabytes
            memory_stats.append(peak_memory)  # Store the peak memory usage



            # Measures time
            duration = start_event.elapsed_time(end_event)
            durations.append(duration)


    average_memory = sum(memory_stats) / len(memory_stats)
    print(f'Average peak memory {average_memory:.2f} MB')
    avg_duration = sum(durations) / len(durations)
    print(f"Average inference time: {avg_duration} ms")




def main():

    

    model = create_seg_model('b0', 'cityscapes', weight_url=None)
    model = nn.DataParallel(model)
    print_model_size(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'{pytorch_total_params} params')    
    print(f'running... with {torch.cuda.device_count()} GPUs, cur device {torch.cuda.current_device()}')


    
    nums = [1, 2, 4, 8, 16]
    for num in nums:
        input_tensor = torch.randn(num, 3, 1216, 1920)  # Example input tensor
        print(f'\nbatch size {num}')
        measure_inference_time(model, input_tensor)


if __name__ == "__main__":
    main()
