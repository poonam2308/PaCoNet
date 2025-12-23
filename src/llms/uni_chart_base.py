import sys
from pathlib import Path

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re

model_name = "ahmed-masry/unichart-chartqa-960"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up from this file
sys.path.insert(0, project_root)

project_root = Path(project_root)

image_path = project_root / "data/synthetic_plots/testing/images_100/image_1_crop_1.png"
input_prompt = "<chartqa> What are the coordinates of different colors of lines ? <s_answer>"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image = Image.open(image_path).convert("RGB")
decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=0,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = sequence.split("<s_answer>")[1].strip()
print(sequence)
