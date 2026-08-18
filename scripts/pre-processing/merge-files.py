

import os
import os
import re
import shutil

def merge_part_files(input_folder: str, output_folder: str) -> None:
    """
    Merges files matching the pattern <file-name>_part_00NN-out.jsonl
    into single combined files named <file-name>_out.jsonl.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Match pattern: <base_name>_part_<digits>-out.jsonl
    pattern = re.compile(r"^(.*?)(?:_part_\d+)(-out\.jsonl)$")
    
    # Group files by target output filename
    groups = {}
    for filename in os.listdir(input_folder):
        match = pattern.match(filename)
        if match:
            base_name, suffix = match.groups()
            target_filename = f"{base_name}{suffix}"
            groups.setdefault(target_filename, []).append(filename)
            
    # Process each grouped file set
    for target_filename, part_files in groups.items():
        # Sort part files numerically to preserve correct order
        part_files.sort()
        
        output_filepath = os.path.join(output_folder, target_filename)
        
        with open(output_filepath, "wb") as outfile:
            for part_file in part_files:
                part_filepath = os.path.join(input_folder, part_file)
                with open(part_filepath, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)

if __name__ == "__main__":
    input_folder = "C:\\Users\\Milind\\MyData\\git\\datasets\\content-moderation-dataset\\Aegis-AI-Content-Safety-Dataset-2.0\\gemini-output"
    output_folder = "C:\\Users\\Milind\\MyData\\git\\datasets\\content-moderation-dataset\\Aegis-AI-Content-Safety-Dataset-2.0\\merged-output"

    print("\n*** Merge files together ***")

    merge_part_files(input_folder, output_folder)
    
    
    print("\n*** Processing complete ***")
    