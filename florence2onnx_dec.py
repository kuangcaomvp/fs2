import os
from unittest.mock import patch

import onnx
import onnxsim
import torch
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa


class florence2DecWrapper(torch.nn.Module):
    def __init__(self, decoder, lm_head, final_logits_bias):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias

    # B L  #  B H W  # B H
    def forward(self, input, encoder_hidden_states, encoder_attention_mask):
        input_shape = input.shape
        inputs_embeds = self.decoder.embed_tokens(input)
        encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            encoder_attention_mask,
            inputs_embeds.dtype,
            tgt_len=input_shape[-1],
        )
        positions = self.decoder.embed_positions(input, 0)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.decoder.layernorm_embedding(hidden_states)

        for idx, decoder_layer in enumerate(self.decoder.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                output_attentions=None,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        return lm_logits

def build_onnx_engine():
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
    language_model = model.language_model
    decoder_input_ids = torch.ones((3,17),dtype=torch.int64,device=device)
    encoder_hidden_states = torch.ones((3,590,768),dtype=dtype,device=device)
    encoder_attention_mask= torch.ones((3,590),dtype=dtype,device=device)
    dec_wrap = florence2DecWrapper(language_model.get_decoder(),language_model.get_output_embeddings(),
                                   language_model.final_logits_bias)
    onnx_dir = 'output/onnx'
    os.makedirs(onnx_dir, exist_ok=True)

    torch.onnx.export(dec_wrap, (decoder_input_ids,
                                 encoder_hidden_states,
                                 encoder_attention_mask),
                      onnx_dir + '/florence2_dec_model.onnx',
                      verbose=False,
                      opset_version=16,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['decoder_input_ids', 'encoder_hidden_states', 'encoder_attention_mask'],
                      output_names=['lm_logits'],
                      dynamic_axes={
                          'decoder_input_ids': {
                            #   0: 'batch',
                              1: 'l'
                          },
                          'encoder_hidden_states': {
                              1: 'width',
                          },
                          'encoder_attention_mask': {
                              1: 'width',
                          }
                      },
                    #   dynamic_axes=None
                      )

    model_onnx = onnx.load(onnx_dir + '/florence2_dec_model.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    try:

        model_onnx, check = onnxsim.simplify(
            model_onnx)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_dir + '/florence2_dec_model.onnx')
    except Exception as e:
        print(f' simplifier failure: {e}')



build_onnx_engine()
