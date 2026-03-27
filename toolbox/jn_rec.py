import os
from .utils.jn import run
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import re
import csv

def JERSEY_NUMBER_RECOGNITION(query=None, material=[]):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    cudnn.benchmark = True

    # Check if we got empty material
    if not material:
        return "No valid images found for jersey number recognition."

    # Parse what kind of material we got
    image_paths = []

    # Process the material argument
    for path in material:
        # It might be a directory
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(root, file))
        # Or it might be a direct list of image paths
        else:
            if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(path)

    try:
        image_paths.sort(key=lambda x: int(re.search(r'(\d+)\.(jpg|jpeg|png)', x.lower()).group(1)))
    except:
        image_paths.sort()

    if not image_paths:
        return "No valid images found for jersey number recognition."

    # Apply FPS sampling down to avoid passing too many frames to FlashAttention and OOM
    # If there's a lot of frames, let's sample it.
    # Usually 25 FPS video, taking every 5th frame = 5 FPS
    stride = 1
    if len(image_paths) > 20:
        stride = len(image_paths) // 20
    elif len(image_paths) > 10:
        stride = 2

    sampled_paths = image_paths[::stride] if stride > 1 else image_paths

    image_list = []
    for path in sampled_paths:
        try:
            img = Image.open(path).convert('RGB')
            image_list.append(img)
        except Exception as e:
            print(f"Error: Could not open image {path}: {e}")

    if not image_list:
        return "Error: Failed to load any valid images."

    model_path = "/home/zhaosiyao/SoccerAgent/toolbox/utils/weights/legibility_resnet34_soccer_20240215.pth" # Replace with your actual model path
    qwen_path = "/data/zhaosiyao/model/Qwen-Qwen2.5-VL-7B-Instruct/"
    ans, result = run(device, image_list, model_path, qwen_path, threshold=0.5)
    ans = -1 if ans == None else ans
    ans = f"The jersey number in the pictures is {ans}."
    return ans
