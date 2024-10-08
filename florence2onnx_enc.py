import os
from unittest.mock import patch

import onnx
import onnxsim
import torch
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa


def _merge_input_ids_with_image_features(
        image_features, inputs_embeds
):
    batch_size, image_token_length = image_features.size()[:-1]
    device = image_features.device
    image_attention_mask = torch.ones(batch_size, image_token_length, dtype=inputs_embeds.dtype, device=device)

    # task_prefix_embeds: [batch_size, padded_context_length, hidden_size]
    # task_prefix_attention_mask: [batch_size, context_length]
    if inputs_embeds is None:
        return image_features, image_attention_mask

    task_prefix_embeds = inputs_embeds
    task_prefix_attention_mask = torch.ones(batch_size, task_prefix_embeds.size(1), dtype=inputs_embeds.dtype,
                                            device=device)

    if len(task_prefix_attention_mask.shape) == 3:
        task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

    # concat [image embeds, task prefix embeds]
    inputs_embeds = torch.cat([image_features, task_prefix_embeds], dim=1)
    attention_mask = torch.cat([image_attention_mask, task_prefix_attention_mask], dim=1)

    return inputs_embeds, attention_mask


class florence2EncWrapper(torch.nn.Module):

    def __init__(self, language_model, vismodel, image_pos_embed, visual_temporal_embed, image_projection,
                 image_proj_norm):
        super().__init__()
        # 语言模型词向量
        self.embeddings = language_model.get_input_embeddings()
        # 编码
        self.encoder = language_model.get_encoder()
        self.enc_embed = self.encoder.embed_positions

        # 图像特征提取模型
        self.vismodel = vismodel
        self.image_pos_embed = image_pos_embed
        self.visual_temporal_embed = visual_temporal_embed
        self.image_proj_norm = image_proj_norm
        self.image_projection = image_projection

    def forward(self, input_ids, pixel_values):
        batch_size, C, H, W = pixel_values.shape
        T = 1
        # 视觉模型提取特征
        x = self.vismodel.forward_features_unpool(pixel_values)

        # 添加位置编码
        x = x.view(batch_size * T, -1, x.shape[-1])
        num_tokens = x.shape[-2]
        h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)

        x = x.view(batch_size * T, h, w, x.shape[-1])
        pos_embed = self.image_pos_embed(x)
        x = x + pos_embed
        x = x.view(batch_size, T * h * w, x.shape[-1])

        # 转序列编码
        visual_temporal_embed = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
        x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        new_x = []
        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        new_x.append(spatial_avg_pool_x)
        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        new_x.append(temporal_avg_pool_x)
        x = torch.cat(new_x, dim=1)

        x = x @ self.image_projection
        x = self.image_proj_norm(x)

        inputs_embeds = self.embeddings(input_ids)
        # # 特征融合
        inputs_embeds, attention_mask = _merge_input_ids_with_image_features(x, inputs_embeds)

        input = inputs_embeds[:, :, -1]
        embed_pos = self.enc_embed(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.encoder.layernorm_embedding(hidden_states)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.encoder.dropout, training=False)

        attention_mask_exp = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
        for idx, encoder_layer in enumerate(self.encoder.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask_exp,
                layer_head_mask=None,
                output_attentions=None,
            )

            hidden_states = layer_outputs[0]

        return (hidden_states, attention_mask)


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
    wrapper = florence2EncWrapper(model.language_model, model.vision_tower, model.image_pos_embed,
                                  model.visual_temporal_embed, model.image_projection, model.image_proj_norm)

    input_ids = torch.ones((1, 13), dtype=torch.int64, device=device)
    pixel_values = torch.ones((1, 3, 768, 768), dtype=dtype, device=device)
    onnx_dir = 'output/onnx'
    os.makedirs(onnx_dir, exist_ok=True)
    torch.onnx.export(wrapper, (input_ids, pixel_values),
                      onnx_dir + '/florence2_enc_model.onnx',
                      verbose=False,
                      opset_version=16,
                      training=torch.onnx.TrainingMode.EVAL,
                      keep_initializers_as_inputs=False,
                      do_constant_folding=True,
                      input_names=['input', 'pixel'],
                      output_names=['encoder_outputs', 'attention_mask'],
                      dynamic_axes={
                          'input': {
                              #   0: 'batch',
                              1: 'length'
                          },
                          #   'pixel': {
                          #       0: 'batch',

                          #   }
                      },
                      #   dynamic_axes=None
                      )

    model_onnx = onnx.load(onnx_dir + '/florence2_enc_model.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    try:

        model_onnx, check = onnxsim.simplify(
            model_onnx)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_dir + '/florence2_enc_model.onnx')
    except Exception as e:
        print(f' simplifier failure: {e}')


build_onnx_engine()
