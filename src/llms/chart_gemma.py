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


#
# import os
# import sys
# from pathlib import Path
#
# from PIL import Image
# from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
# import torch
#
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# sys.path.insert(0, project_root)
# project_root = Path(project_root)
#
# image_path = project_root / "data/synthetic_plots/testing/images_100/image_1_crop_1.png"
#
# # ---- Chaining config ----
# num_steps = 3
#
# # Step 1 instruction
# base_question = "What are the coordinates of different colors of lines?"
#
# # How to build the next prompt:
# # Option A: append previous answer as context (recommended)
# def make_next_prompt(step_idx: int, prev_answer: str) -> str:
#     return (
#         f"Task: {base_question}\n"
#         f"Previous answer (step {step_idx}): {prev_answer}\n\n"
#         "Now refine/correct the answer if needed, and output ONLY the final coordinates per color."
#     )
#
# # Option B: replace prompt entirely with previous output (sometimes useful for formatting passes)
# # def make_next_prompt(step_idx: int, prev_answer: str) -> str:
# #     return (
# #         "Convert the following into clean JSON with keys as colors and values as coordinate arrays:\n"
# #         f"{prev_answer}"
# #     )
#
# # ---- Load model/processor ----
# model = PaliGemmaForConditionalGeneration.from_pretrained(
#     "ahmed-masry/chartgemma",
#     torch_dtype=torch.float16
# )
# processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
#
# # ---- Load image once ----
# image = Image.open(image_path).convert("RGB")
#
# prev = None
# for step in range(1, num_steps + 1):
#     if step == 1:
#         input_text = base_question
#     else:
#         input_text = make_next_prompt(step - 1, prev)
#
#     inputs = processor(text=input_text, images=image, return_tensors="pt")
#     prompt_length = inputs["input_ids"].shape[1]
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#
#     with torch.inference_mode():
#         generate_ids = model.generate(**inputs, max_new_tokens=256)
#
#     out = processor.batch_decode(
#         generate_ids[:, prompt_length:],
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )[0]
#
#     print(f"\n--- STEP {step} PROMPT ---\n{input_text}")
#     print(f"\n--- STEP {step} OUTPUT ---\n{out}")
#
#     prev = out  # <- chaining happens here
#
# print("\n=== FINAL OUTPUT ===\n", prev)
