# List video paths to json
import os
import json

def create_video_paths_json(data_dir='mini-sample', output_json='example_video_paths.json'):
    """
    Scans a directory for video files and saves their paths to a JSON file.

    Args:
        data_dir (str): The directory containing the video files.
        output_json (str): The path to the output JSON file.
    """
    try:
        # Get all file names in the data directory
        video_files = os.listdir(data_dir)

        # Create full paths for each file
        video_paths = [os.path.join(data_dir, f) for f in video_files if os.path.isfile(os.path.join(data_dir, f))]

        # Create the dictionary to be saved as JSON
        output_data = {"video_paths": video_paths}

        # Write the data to the JSON file
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Successfully created {output_json} with {len(video_paths)} video paths.")

    except FileNotFoundError:
        print(f"Error: The directory '{data_dir}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    create_video_paths_json()