import os
import sys
from pathlib import Path

from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up from this file
sys.path.insert(0, project_root)

project_root = Path(project_root)

image_path = project_root / "data/synthetic_plots/testing/images_100/image_1_crop_1.png"
input_text ="What are the coordinates of different colors of lines?"

# Load Model
model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Process Inputs
image = Image.open(image_path).convert('RGB')
inputs = processor(text=input_text, images=image, return_tensors="pt")
prompt_length = inputs['input_ids'].shape[1]
inputs = {k: v.to(device) for k, v in inputs.items()}


# Generate
generate_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)
