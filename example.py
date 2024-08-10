import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import time

def fixed_get_imports(filename: str):
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("Florence-2-base",torch_dtype=torch_dtype,
                                             trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("Florence-2-base", trust_remote_code=True)

image = Image.open('car.jpg')


def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt,
                                                      image_size=(image.width, image.height))


    print(parsed_answer)


# prompt = "<CAPTION>"
# prompt = "<DETAILED_CAPTION>"
prompt = "<OD>"
# prompt = "<DENSE_REGION_CAPTION>"
# prompt = "<REGION_PROPOSAL>"
# prompt = "<OCR>"
# prompt = "<OCR_WITH_REGION>"
for i in range(10):
 t1 = time.time()
 run_example(prompt)
 print(time.time() - t1)

t2 = time.time()
print(t2 - t1)
