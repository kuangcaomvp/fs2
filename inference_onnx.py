import time

import onnxruntime
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList, NoRepeatNGramLogitsProcessor, \
    ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria, EosTokenCriteria
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_bbox(image, data):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    plt.show()

def prepare_inputs_for_generation(
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
):
    # cut decoder_input_ids if past_key_values is used
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]

        # Some generation methods already pass only the last input ID
        if decoder_input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = decoder_input_ids.shape[1] - 1

        decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

    return {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
    }

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

    dec_session = onnxruntime.InferenceSession(r'output\onnx\florence2_dec_model.onnx', providers=providers)

    class florence2Wrapper(torch.nn.Module):

        def __init__(self,
                     tokenizer,
                     post_processor):
            super().__init__()
            # 语言模型
            self.g = GenerationMixin()

            # 语言模型配置文件
            with open('config/opt.yaml', errors='ignore') as f:
                self.GenerationConfig = yaml.safe_load(f)  # language_model.generation_config

            self.decoder_input_id = self.GenerationConfig['decoder_start_token_id']
            self.decoder_input_id = torch.tensor([[self.decoder_input_id]], dtype=torch.long,device=device)

            # 语言模型decoder 前处理逻辑
            self.prepared_logits_processor = LogitsProcessorList()
            self.prepared_logits_processor.append(
                NoRepeatNGramLogitsProcessor(self.GenerationConfig['no_repeat_ngram_size']))  # 3
            self.prepared_logits_processor.append(
                ForcedBOSTokenLogitsProcessor(self.GenerationConfig['forced_bos_token_id']))  # 0
            self.prepared_logits_processor.append(
                ForcedEOSTokenLogitsProcessor(
                    self.GenerationConfig['max_length'],  # 20
                    self.GenerationConfig['forced_eos_token_id'],  # 2
                    device=device,
                )
            )

            # 语言模型decoder 后处理逻辑
            self.prepared_stopping_criteria = StoppingCriteriaList()
            self.prepared_stopping_criteria.append(
                MaxLengthCriteria(
                    max_length=self.GenerationConfig['max_length'],  # 20
                    max_position_embeddings=1024,  # 1024
                )
            )
            self.prepared_stopping_criteria.append(EosTokenCriteria(eos_token_id=self.GenerationConfig['eos_token_id']))

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
                num_beams=self.GenerationConfig['num_beams'],  # 3
                device=device,
                length_penalty=self.GenerationConfig['length_penalty'],  # 1.0
                do_early_stopping=self.GenerationConfig['early_stopping'],  # True
                num_beam_hyps_to_keep=self.GenerationConfig['num_return_sequences'],  # 1
                max_length=self.GenerationConfig['max_length'],  # 20
            )
            decoder_inputs, model_kwargs = self.g._expand_inputs_for_generation(
                input_ids=self.decoder_input_id,
                expand_size=self.GenerationConfig['num_beams'],
                is_encoder_decoder=True,
                **model_kwargs,
            )

            pad_token_id = torch.tensor(self.GenerationConfig['_pad_token_tensor'], device=device)
            eos_token_id = torch.tensor([self.GenerationConfig['_eos_token_tensor']], device=device)
            output_scores = self.GenerationConfig['output_scores']
            return_dict_in_generate = self.GenerationConfig['return_dict_in_generate']
            batch_size = len(beam_scorer._beam_hyps)
            num_beams = beam_scorer.num_beams
            batch_beam_size, cur_len = decoder_inputs.shape
            model_kwargs = self.g._get_initial_cache_position(decoder_inputs, model_kwargs)
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))
            this_peer_finished = False
            decoder_prompt_len = decoder_inputs.shape[-1]  # record the prompt length of decoder
            beam_indices = (
                tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
            )
            while not this_peer_finished:
                model_inputs = prepare_inputs_for_generation(decoder_inputs, **model_kwargs)

                lm_logits, next_token_scores = dec_session.run(
                    ['lm_logits', 'next_token_scores'],
                    input_feed={
                        'decoder_input_ids': model_inputs['decoder_input_ids'].cpu().numpy(),
                        'encoder_outputs': model_inputs['encoder_outputs'][0].cpu().numpy(),
                        'attention_mask': model_inputs['attention_mask'].cpu().numpy()

                    })
                lm_logits = torch.tensor(lm_logits, device=device)
                next_token_scores = torch.tensor(next_token_scores, device=device)
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
                model_kwargs = self.g._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=True,
                )
                del outputs
                if model_kwargs.get("past_key_values", None) is not None:
                    model_kwargs["past_key_values"] = self.g._temporary_reorder_cache(
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
            return final_answer


    wrapper = florence2Wrapper(processor.tokenizer, processor.post_processor)
    # for i in range(10):
    t1 = time.time()
    out= wrapper(encoder_outputs, inputs_embeds, attention_mask)
    print(out)
    print(time.time() - t1)
    plot_bbox(image, out)


build_onnx_engine()
