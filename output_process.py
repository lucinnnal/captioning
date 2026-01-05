import json

def convert_json_to_jsonl(input_file_path, output_file_path):
    """
    Reads a JSON file where keys are video names and values are captions,
    and converts it into a JSONL format.

    Each line in the output file will be a JSON object with "video_name"
    and "caption" keys.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}")
        return

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for video_name, caption in data.items():
            output_obj = {"video_name": video_name, "caption": caption}
            outfile.write(json.dumps(output_obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_json_file = "output_captions.json"
    output_jsonl_file = "internVL-30B-3A_captions.jsonl"
    convert_json_to_jsonl(input_json_file, output_jsonl_file)
    print(f"Conversion complete. Output written to {output_jsonl_file}")