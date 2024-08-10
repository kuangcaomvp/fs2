import numpy as np
import onnxruntime
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import os
import onnx
from transformers.generation.logits_process import LogitsProcessorList, NoRepeatNGramLogitsProcessor, \
    ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria, EosTokenCriteria
from transformers.generation.beam_search import BeamSearchScorer
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import time

def build_onnx_engine():
    processor = AutoProcessor.from_pretrained("Florence-2-base", trust_remote_code=True)

    image = Image.open('car.jpg')  # Image.open('car.jpg')  # Image.new('RGB', [768, 768])  # dummy image
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    inputs = processor(text="<OD>", images=image, return_tensors="pt").to(device, dtype)

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else [
        'CPUExecutionProvider']
    enc_session = onnxruntime.InferenceSession(r'output\onnx\florence2_enc_model.onnx', providers=providers)
    encoder_outputs, inputs_embeds, attention_mask = enc_session.run(
        ['encoder_outputs', 'inputs_embeds', 'attention_mask'],
        input_feed={'input': input_ids.cpu().numpy(),
                    'pixel': pixel_values.cpu().numpy()})
    encoder_outputs = torch.tensor(encoder_outputs, device=device)
    inputs_embeds = torch.tensor(inputs_embeds, device=device)
    attention_mask = torch.tensor(attention_mask, device=device)

    class florence2DecWrapper(torch.nn.Module):
        def __init__(self, decoder, lm_head, final_logits_bias):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.final_logits_bias = final_logits_bias

        # B L  #  B H W  # B H
        def forward(self, decoder_input_ids, encoder_outputs, attention_mask):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
            lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
            next_token_logits = lm_logits[:, -1, :].clone()
            next_token_scores = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            return lm_logits,  next_token_scores

    class florence2Wrapper(torch.nn.Module):

        def __init__(self, language_model,
                     tokenizer,
                     post_processor):
            super().__init__()
            # 语言模型
            self.language_model = language_model
            self.dec_wrap = florence2DecWrapper(language_model.get_decoder(), language_model.get_output_embeddings(),
                                                language_model.final_logits_bias)

            # 语言模型配置文件
            self.GenerationConfig = language_model.generation_config
            self.language_model._prepare_special_tokens(self.GenerationConfig, False, device=device)
            self.decoder_input_id = self.GenerationConfig.decoder_start_token_id
            self.decoder_input_id = torch.tensor([[self.decoder_input_id]], dtype=torch.long,device=device)

            # 语言模型decoder 前处理逻辑
            self.prepared_logits_processor = LogitsProcessorList()
            self.prepared_logits_processor.append(
                NoRepeatNGramLogitsProcessor(self.GenerationConfig.no_repeat_ngram_size))  # 3
            self.prepared_logits_processor.append(
                ForcedBOSTokenLogitsProcessor(self.GenerationConfig.forced_bos_token_id))  # 0
            self.prepared_logits_processor.append(
                ForcedEOSTokenLogitsProcessor(
                    self.GenerationConfig.max_length,  # 20
                    self.GenerationConfig.forced_eos_token_id,  # 2
                    device=device,
                )
            )

            # 语言模型decoder 后处理逻辑
            self.prepared_stopping_criteria = StoppingCriteriaList()
            self.prepared_stopping_criteria.append(
                MaxLengthCriteria(
                    max_length=self.GenerationConfig.max_length,  # 20
                    max_position_embeddings=1024,  # 1024
                )
            )
            self.prepared_stopping_criteria.append(EosTokenCriteria(eos_token_id=self.GenerationConfig.eos_token_id))

            self.tokenizer = tokenizer
            self.post_processor = post_processor
            self.task_answer_post_processing_type = 'description_with_bboxes'

        def forward(self, encoder_outputs, inputs_embeds, attention_mask):
            batch_size = inputs_embeds.shape[0]
            model_kwargs = {
                'inputs_embeds': inputs_embeds,
                'use_cache': False,
                'attention_mask': attention_mask,
                'encoder_outputs': BaseModelOutput(
                    last_hidden_state=encoder_outputs, hidden_states=None, attentions=None
                )
            }
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=self.GenerationConfig.num_beams,  # 3
                device=device,
                length_penalty=self.GenerationConfig.length_penalty,  # 1.0
                do_early_stopping=self.GenerationConfig.early_stopping,  # True
                num_beam_hyps_to_keep=self.GenerationConfig.num_return_sequences,  # 1
                max_length=self.GenerationConfig.max_length,  # 20
            )
            decoder_inputs, model_kwargs = self.language_model._expand_inputs_for_generation(
                input_ids=self.decoder_input_id,
                expand_size=self.GenerationConfig.num_beams,
                is_encoder_decoder=True,
                **model_kwargs,
            )

            pad_token_id = self.GenerationConfig._pad_token_tensor
            eos_token_id = self.GenerationConfig._eos_token_tensor
            output_scores = self.GenerationConfig.output_scores
            return_dict_in_generate = self.GenerationConfig.return_dict_in_generate
            batch_size = len(beam_scorer._beam_hyps)
            num_beams = beam_scorer.num_beams
            batch_beam_size, cur_len = decoder_inputs.shape
            model_kwargs = self.language_model._get_initial_cache_position(decoder_inputs, model_kwargs)
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))
            this_peer_finished = False
            decoder_prompt_len = decoder_inputs.shape[-1]  # record the prompt length of decoder
            beam_indices = (
                tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
            )
            while not this_peer_finished:
                model_inputs = self.language_model.prepare_inputs_for_generation(decoder_inputs, **model_kwargs)
                lm_logits,  next_token_scores = self.dec_wrap(model_inputs['decoder_input_ids'],
                                                                              model_inputs['encoder_outputs'][0],
                                                                              model_inputs['attention_mask'])

                outputs = Seq2SeqLMOutput(
                    logits=lm_logits,
                    encoder_last_hidden_state=model_inputs['encoder_outputs'][0],
                )
                next_token_scores_processed = self.prepared_logits_processor(decoder_inputs, next_token_scores)
                next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                    next_token_scores_processed
                )
                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
                n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
                n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )
                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                beam_outputs = beam_scorer.process(
                    decoder_inputs,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=beam_indices,
                    decoder_prompt_len=decoder_prompt_len,
                )

                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                decoder_inputs = torch.cat([decoder_inputs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                model_kwargs = self.language_model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.language_model.config.is_encoder_decoder,
                )
                del outputs
                if model_kwargs.get("past_key_values", None) is not None:
                    model_kwargs["past_key_values"] = self.language_model._temporary_reorder_cache(
                        model_kwargs["past_key_values"], beam_idx
                    )
                cur_len = cur_len + 1

                if beam_scorer.is_done or all(self.prepared_stopping_criteria(decoder_inputs, None)):
                    this_peer_finished = True

            sequence_outputs = beam_scorer.finalize(
                decoder_inputs,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=self.prepared_stopping_criteria.max_length,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            generated_ids = sequence_outputs["sequences"]

            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
            task_answer = self.post_processor(
                text=generated_text,
                image_size=(image.width, image.height),
                parse_tasks=self.task_answer_post_processing_type,
            )[self.task_answer_post_processing_type]
            od_instances = task_answer
            bboxes_od = [_od_instance['bbox'] for _od_instance in od_instances]

            labels_od = [str(_od_instance['cat_name']) for _od_instance in od_instances]
            final_answer = {'bboxes': bboxes_od, 'labels': labels_od}
            return final_answer, model_inputs

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
    wrapper = florence2Wrapper(model.language_model, processor.tokenizer, processor.post_processor)
    for i in range(10):
        t1 = time.time()
        out, a = wrapper(encoder_outputs, inputs_embeds, attention_mask)
        print(out)
        print(time.time() - t1)

    enc_wrap = florence2DecWrapper(model.language_model.get_decoder(), model.language_model.get_output_embeddings(),
                                   model.language_model.final_logits_bias)
    onnx_dir = 'output/onnx'
    os.makedirs(onnx_dir, exist_ok=True)

    torch.onnx.export(enc_wrap, (a['decoder_input_ids'],
                                 a['encoder_outputs'][0],
                                 a['attention_mask']),
                      onnx_dir + '/florence2_dec_model.onnx',
                      verbose=False,
                      opset_version=17,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['decoder_input_ids', 'encoder_outputs', 'attention_mask'],
                      output_names=['lm_logits', 'next_token_scores'],
                      dynamic_axes={
                          'decoder_input_ids': {
                              0: 'batch',
                              1: 'length'
                          },
                          'encoder_outputs': {
                              0: 'batch',
                              1: 'height',
                              2: 'width'
                          },
                          'attention_mask': {
                              0: 'batch',
                              1: 'height'
                          },
                          'lm_logits':{
                              0: 'batch',
                              1: 'c'
                          }
                      }
                      )

    model_onnx = onnx.load(onnx_dir + '/florence2_dec_model.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model


build_onnx_engine()
