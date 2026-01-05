import json
import os

def convert_captions(input_file, output_file, base_path):
    """
    Converts a JSONL-like file with concatenated JSON objects to a single JSON array file.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        base_path (str): The base path for the video URLs in the output.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # The file contains multiple pretty-printed JSON objects concatenated.
    # We can't just split by newline.
    # A robust way is to use a JSONDecoder to decode multiple objects.
    decoder = json.JSONDecoder()
    objs = []
    pos = 0
    content = content.strip()
    while pos < len(content):
        try:
            obj, size = decoder.raw_decode(content[pos:])
            objs.append(obj)
            pos += size
            # Skip whitespace and newlines between objects
            while pos < len(content) and content[pos].isspace():
                pos += 1
        except json.JSONDecodeError as e:
            print(f"Stopped at position {pos} with error: {e}")
            # Handle trailing characters or malformed json
            break


    converted_data = []
    for item in objs:
        if 'data' in item and 'video' in item['data'] and 'sentences' in item['data']:
            video_path = item['data']['video']
            video_filename = os.path.basename(video_path)
            new_video_url = "/data/local-files/?d=" + os.path.join(base_path, video_filename)

            caption = " ".join(sentence['text'] for sentence in item['data']['sentences'])

            converted_data.append({
                "video_url": new_video_url,
                "caption": caption
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    INPUT_FILENAME = "internvl3_5_30B_A3B.jsonl"
    OUTPUT_FILENAME = "internvl3_5_30B_A3B_converted.json"
    BASE_VIDEO_PATH = "/Users/kipyokim/mini-sample"

    convert_captions(INPUT_FILENAME, OUTPUT_FILENAME, BASE_VIDEO_PATH)

    print(f"Conversion complete. Output written to {OUTPUT_FILENAME}")
