import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tensorrt as trt
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList, NoRepeatNGramLogitsProcessor, \
    ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria, EosTokenCriteria

from onnx_trt.backend_infer import TensorRTBackendRep as backend


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


def load_trt_engine(path: str) -> trt.ICudaEngine:
    """Deserialize TensorRT engine from disk.

    Arguments:
        path (str): disk path to read the engine

    Returns:
        tensorrt.ICudaEngine: the TensorRT engine loaded from disk
    """

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine


class florence2Wrapper(torch.nn.Module):

    def __init__(self, device):
        super().__init__()

        # 语言模型配置文件
        with open('config/opt.yaml', errors='ignore') as f:
            self.GenerationConfig = yaml.safe_load(f)  # language_model.generation_config
        providers = ['CUDAExecutionProvider',
                     'CPUExecutionProvider'] if torch.cuda.is_available() else [
            'CPUExecutionProvider']

        self.enc_session = backend('output/onnx/florence2_enc_model.engine', device_id=0)

        self.dec_session = backend('output/onnx/florence2_dec_model.engine', device_id=0)

        self.device = device
        self.decoder_input_id = self.GenerationConfig['decoder_start_token_id']
        self.decoder_input_id = torch.tensor([[self.decoder_input_id]], dtype=torch.long, device=device)

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

        self.beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=self.GenerationConfig['num_beams'],  # 3
            device=self.device,
            length_penalty=self.GenerationConfig['length_penalty'],  # 1.0
            do_early_stopping=self.GenerationConfig['early_stopping'],  # True
            num_beam_hyps_to_keep=self.GenerationConfig['num_return_sequences'],  # 1
            max_length=self.GenerationConfig['max_length'],  # 20
        )

    def forward(self, input_ids, pixel_values):

        ot = self.enc_session.run([input_ids.cpu().numpy(),
                                   pixel_values.cpu().numpy()])
        encoder_outputs = torch.tensor(ot.encoder_outputs, device=self.device)
        attention_mask = torch.tensor(ot.attention_mask, device=self.device)

        expand_size = self.GenerationConfig['num_beams']
        decoder_inputs = self.decoder_input_id.repeat_interleave(expand_size, dim=0)
        encoder_attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        last_hidden_state = encoder_outputs.repeat_interleave(expand_size, dim=0)

        pad_token_id = torch.tensor(self.GenerationConfig['_pad_token_tensor'], device=self.device)
        eos_token_id = torch.tensor([self.GenerationConfig['_eos_token_tensor']], device=self.device)

        batch_size = len(self.beam_scorer._beam_hyps)
        num_beams = self.beam_scorer.num_beams
        batch_beam_size, cur_len = decoder_inputs.shape
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        this_peer_finished = False
        decoder_prompt_len = decoder_inputs.shape[-1]  # record the prompt length of decoder
        beam_indices = None
        while not this_peer_finished:

            dot = self.dec_session.run(
                [decoder_inputs.cpu().numpy(), last_hidden_state.cpu().numpy(), encoder_attention_mask.cpu().numpy()])
            lm_logits = torch.tensor(dot.lm_logits, device=self.device)

            next_token_logits = lm_logits[:, -1, :].clone()
            next_token_scores = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

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

            beam_outputs = self.beam_scorer.process(
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

            cur_len = cur_len + 1

            if self.beam_scorer.is_done or all(self.prepared_stopping_criteria(decoder_inputs, None)):
                this_peer_finished = True

        sequence_outputs = self.beam_scorer.finalize(
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

        return generated_ids


def build_onnx_engine():
    processor = AutoProcessor.from_pretrained("Florence-2-base", trust_remote_code=True)

    image = Image.open('dogs.jpg')  # Image.open('car.jpg')  # Image.new('RGB', [768, 768])  # dummy image
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    inputs = processor(text="<OD>", images=image, return_tensors="pt").to(device, dtype)

    # inputs = processor(text="<OPEN_VOCABULARY_DETECTION>black dog", images=image, return_tensors="pt").to(device, dtype)

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    wrapper = florence2Wrapper(device)
    for i in range(10):
        t1 = time.time()
        out = wrapper(input_ids, pixel_values)
        print(time.time() - t1)

    generated_text = processor.tokenizer.batch_decode(out, skip_special_tokens=False)[0]

    # task_answer = processor.post_processor(
    #     text=generated_text,
    #     image_size=(image.width, image.height),
    #     parse_tasks='description_with_bboxes_or_polygons',
    # )['description_with_bboxes_or_polygons']

    task_answer = processor.post_processor(
        text=generated_text,
        image_size=(image.width, image.height),
        parse_tasks='description_with_bboxes',
    )['description_with_bboxes']
    od_instances = task_answer
    bboxes_od = [_od_instance['bbox'] for _od_instance in od_instances]
    labels_od = [str(_od_instance['cat_name']) for _od_instance in od_instances]
    final_answer = {'bboxes': bboxes_od, 'labels': labels_od}
    print(final_answer)

    plot_bbox(image, final_answer)


build_onnx_engine()
