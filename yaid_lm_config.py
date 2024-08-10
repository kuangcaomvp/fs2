from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import time
import yaml

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
def fixed_get_imports(filename: str):
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("Florence-2-base", torch_dtype=dtype,
                                                 trust_remote_code=True).to(device)
model.eval()

lm = model.language_model
GenerationConfig = lm.generation_config
lm._prepare_special_tokens(GenerationConfig, False, device=device)
out = vars(GenerationConfig)
for a in out:
    if isinstance(out[a],torch.Tensor):
        out[a] = int(out[a])
with open('config/opt.yaml', 'w') as f:
    yaml.safe_dump(out, f, sort_keys=False)
