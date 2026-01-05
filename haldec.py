import math
import json
import torch
import argparse
import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import InterpolationMode

# Utils
# Each frame to tensor
def build_transform(input_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    MEAN, STD = mean, std
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

# Single frame(image) to 448*448 tiles + Thumbnail
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# Single image preprocess and dynamic porcessing
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Get frame indices to extract (divide video into segments and extract frame in each segment)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

# Video Load + Tensorfy the frames (VideoReader?: )
def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1) 
    max_frame = len(vr) - 1 # last frame index
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num) # max_num=1, does not split into tiles for single image
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0]) # how many tiles for each frame?
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def main(args):
    # Load Model
    max_memory = {
    0: "20GiB",
    1: "20GiB",
    2: "20GiB",
    3: "20GiB",
    4: "20GiB"
    }
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        max_memory=max_memory,
        device_map='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=10, do_sample=False) 
    
    # System Prompt
    if args.use_sys_prompt:
        R1_SYSTEM_PROMPT = args.sys_prompt.strip()
        model.system_message = R1_SYSTEM_PROMPT
    
    EVAL_PROMPT_TEMPLATE = args.eval_prompt_template

    with open(args.output_jsonl_path, 'w') as f_out:
        try:
            with open(args.input_jsonl_path, 'r') as f_in:
                json_content = f_in.read()
            
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(json_content):
                # skip leading whitespace
                i = pos
                while i < len(json_content) and json_content[i].isspace():
                    i += 1
                if i == len(json_content):
                    break # end of file
                pos = i

                data, end_pos = decoder.raw_decode(json_content[pos:])
                pos += end_pos
                
                video_path = data['data']['video']
                sentences = data['data']['sentences']

                pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])

                for sentence_data in sentences:
                    sentence_text = sentence_data['text']
                    
                    question = video_prefix + EVAL_PROMPT_TEMPLATE.format(sentence_text=sentence_text)
                    
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                                   num_patches_list=num_patches_list, history=None, return_history=True)
                    
                    # Clean and parse the response
                    try:
                        binary_decision = int(response.strip())
                    except (ValueError, TypeError):
                        binary_decision = -1 # Indicate an error in parsing
                    
                    result = {
                        "video_path": video_path,
                        "sentence": sentence_text,
                        "id": sentence_data.get('id'),
                        "binary_decision": binary_decision
                    }
                    f_out.write(json.dumps(result) + '\n')
                    print(f"Processed sentence '{sentence_data.get('id')}' for video: {video_path}")

        except Exception as e:
            error_result = {
                "error": str(e),
                "original_file": args.input_jsonl_path
            }
            f_out.write(json.dumps(error_result) + '\n')
            print(f"Error processing file {args.input_jsonl_path}: {e}")

    print(f"Results saved to {args.output_jsonl_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="OpenGVLab/InternVL3_5-8B", help="The name of the model to use.")
    parser.add_argument("--use-sys-prompt", type=bool, default=False, help="Use system prompt?")
    parser.add_argument("--eval-prompt-template", type=str, help="System prompt", default = """You are a strict video-grounded hallucination evaluator. Given a video and a sentence from a caption, you MUST follow these rules:
                        
1) For the given sentence, make a BINARY decision:
- binary = 1 (correct): the sentence is clearly supported by visible evidence in the video.
- binary = 0 (incorrect): the sentence is unsupported, contradicted, or not verifiable from the video.
- If evidence is not explicitly visible, you MUST choose 0. Do NOT infer or guess.

2) Do NOT use world knowledge, common sense, or assumptions. Only judge based on what is observable in the video.

3) The sentence to evaluate is: "{sentence_text}"

4) Output ONLY the binary decision (0 or 1) and nothing else.""")
    parser.add_argument("--sys-prompt", type=str, help="System prompt", default = "You are an AI assistant that rigorously follows this response protocol: 1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags. 2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline. Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.")
    parser.add_argument("--input-jsonl-path", type=str, required=True, help="Path to the input JSONL file containing video paths and sentences.")
    parser.add_argument("--output-jsonl-path", type=str, required=True, help="Path to the output JSONL file to save the results.")
    args = parser.parse_args()
    main(args)