import sys
import subprocess

import torch

def get_gpu_usage():
    """
        Get GPU usage information using nvidia-smi

    Returns:
        str: GPU usage information

    """
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        sys.exit(f"Failed to execute nvidia-smi: {e}")

def is_gpu_free(gpu_id):
    gpu_usage = get_gpu_usage()
    if f' No running processes found' in gpu_usage.split('GPU')[gpu_id + 1]:
        return True
    return False

def setup_device(mydevice: str) -> torch.device:
    mydevice = str(mydevice)
    if not torch.cuda.is_available():

        device = torch.device('cpu')
    else:
        num_gpus = torch.cuda.device_count()

        for i in range(num_gpus):
            print(f"Device {i} name: {torch.cuda.get_device_name(i)}")

        valid_devices = [f'cuda:{i}' for i in range(num_gpus)]
        print(f"Valid GPU references: {valid_devices}")

        if str(mydevice) in valid_devices:
            gpu_id = int(mydevice.split(':')[1])
            if is_gpu_free(gpu_id):
                print(f"Using GPU - {mydevice}")
                device = torch.device(mydevice)
            else:
                print(f"GPU {mydevice} may be currently in use...")
                device = torch.device(mydevice)
        else:   # Define the path to the parameters file
            print(f"Invalid GPU reference: {str(mydevice)} -> Valid references: {valid_devices}. Exiting...")
            sys.exit(1)

    return device