
import pandas as pd
import os
import json

# Define the number of objects per split file
objects_per_file = 2000


def split_file(input_json_file, output_folder):
    
    input_file_name = os.path.basename(input_json_file)
    print(f"Splitting file: {input_json_file}")

    
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError("Input JSON file does not contain a JSON array.")

        print(f"Successfully loaded {len(data)} objects from {input_json_file}")

        file_count = 0
        for i in range(0, len(data), objects_per_file):
            file_count += 1
            chunk = data[i:i + objects_per_file]
            output_file_name = os.path.join(output_folder, f'{input_file_name}_part_{file_count:04d}.json')

            with open(output_file_name, 'w', encoding='utf-8') as outfile:
                json.dump(chunk, outfile, indent=4, ensure_ascii=False)

            print(f"Wrote {len(chunk)} objects to {output_file_name}")

        print(f"*** Successfully split the JSON array into {file_count} files in '{output_folder}'. ***\n")

    except FileNotFoundError:
        print(f"Error: The file '{input_json_file}' was not found. Please check the path.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_json_file}'. Please ensure it's a valid JSON file.")
    except TypeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

def split_files_in_folder(input_folder, output_folder):
    """
    Split all files in a given folder, skipping those without a '.json' extension.

    Args:
        input_folder (str): The path to the folder.
        output_folder (str): The path to splited files.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Folder '{input_folder}' not found or is not a directory.")
        return
    
    print(f"Splitting files in (json files only): {input_folder}")
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isfile(item_path):
            # Check if the file has a .json extension
            if item.lower().endswith('.json'):
                print(f"  - Processing file: {item}")
                split_file(item_path, output_folder)
            else:
                print(f"  - Skipping file: {item}")


if __name__ == "__main__":
    input_folder = "C:\\Users\\Milind\\MyData\\git\\datasets\\content-moderation-dataset\\Aegis-AI-Content-Safety-Dataset-2.0"
    output_folder = "C:\\Users\\Milind\\MyData\\git\\datasets\\content-moderation-dataset\\Aegis-AI-Content-Safety-Dataset-2.0\chunks"
    
    split_files_in_folder(input_folder, output_folder)
    
    print("\n*** Splitting complete ***")
    
    
