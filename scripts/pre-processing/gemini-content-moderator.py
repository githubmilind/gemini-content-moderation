import argparse
import asyncio
import json
import os
import sys
import uuid
from enum import Enum
from typing import List

import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ConfigDict

# --- Pydantic Schemas with Field Aliases matching OpenAI's format ---

class HarmCategoryEnum(str, Enum):
    SAFE = "Safe"
    TOXICITY = "Toxicity or Hate Speech"
    HARASSMENT = "Harassment or Cyberbullying"
    SEXUAL = "Sexually Explicit Content"
    VIOLENCE = "Violence or Dangerous Acts"
    PII = "Personally Identifiable Information"
    ERROR = "Moderation API Error"

class Categories(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sexual: bool = Field(default=False)
    sexual_minors: bool = Field(default=False, alias="sexual/minors")
    harassment: bool = Field(default=False)
    harassment_threatening: bool = Field(default=False, alias="harassment/threatening")
    hate: bool = Field(default=False)
    hate_threatening: bool = Field(default=False, alias="hate/threatening")
    illicit: bool = Field(default=False)
    illicit_violent: bool = Field(default=False, alias="illicit/violent")
    self_harm: bool = Field(default=False, alias="self-harm")
    self_harm_intent: bool = Field(default=False, alias="self-harm/intent")
    self_harm_instructions: bool = Field(default=False, alias="self-harm/instructions")
    violence: bool = Field(default=False)
    violence_graphic: bool = Field(default=False, alias="violence/graphic")


class CategoryScores(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sexual: float = Field(default=0.0)
    sexual_minors: float = Field(default=0.0, alias="sexual/minors")
    harassment: float = Field(default=0.0)
    harassment_threatening: float = Field(default=0.0, alias="harassment/threatening")
    hate: float = Field(default=0.0)
    hate_threatening: float = Field(default=0.0, alias="hate/threatening")
    illicit: float = Field(default=0.0)
    illicit_violent: float = Field(default=0.0, alias="illicit/violent")
    self_harm: float = Field(default=0.0, alias="self-harm")
    self_harm_intent: float = Field(default=0.0, alias="self-harm/intent")
    self_harm_instructions: float = Field(default=0.0, alias="self-harm/instructions")
    violence: float = Field(default=0.0)
    violence_graphic: float = Field(default=0.0, alias="violence/graphic")


class CategoryAppliedInputTypes(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sexual: List[str] = Field(default_factory=list)
    sexual_minors: List[str] = Field(default_factory=list, alias="sexual/minors")
    harassment: List[str] = Field(default_factory=list)
    harassment_threatening: List[str] = Field(default_factory=list, alias="harassment/threatening")
    hate: List[str] = Field(default_factory=list)
    hate_threatening: List[str] = Field(default_factory=list, alias="hate/threatening")
    illicit: List[str] = Field(default_factory=list)
    illicit_violent: List[str] = Field(default_factory=list, alias="illicit/violent")
    self_harm: List[str] = Field(default_factory=list, alias="self-harm")
    self_harm_intent: List[str] = Field(default_factory=list, alias="self-harm/intent")
    self_harm_instructions: List[str] = Field(default_factory=list, alias="self-harm/instructions")
    violence: List[str] = Field(default_factory=list)
    violence_graphic: List[str] = Field(default_factory=list, alias="violence/graphic")


class ModerationResultItem(BaseModel):
    flagged: bool
    categories: Categories
    category_scores: CategoryScores
    category_applied_input_types: CategoryAppliedInputTypes


class OpenAIModerationResponse(BaseModel):
    id: str
    model: str
    results: List[ModerationResultItem]

# New simpler schema for Gemini to populate
class GeminiSimpleModerationResult(BaseModel):
    flagged: bool = Field(
        description="True if the text violates community standards or falls into a harmful category."
    )
    primary_category: HarmCategoryEnum = Field(
        description="The primary harm category matched. Select 'Safe' if the content passes."
    )
    confidence_score: float = Field(
        description="""Confidence score between 0.0 (low confidence) and 1.0 (absolute certainty).
        Set to 0.0 if not flagged or if the model could not determine a specific score."""
    )
    reasoning: str = Field(
        description="A brief, 1-sentence explanation of why the text was flagged or cleared."
    )


# --- Main Moderation Function ---

def moderate_openai_format(text: str) -> dict:
    client = genai.Client()

    # Define system instructions to give Gemini its persona and rules
    system_instruction = (
        "You are an enterprise content moderation system. Analyze the user text objectively. "
        "Ignore spelling attempts to bypass filters (e.g., symbol substitution). Do not moralize, "
        "simply classify the text according to the provided schema instructions."  # Keep this general for simple result
    )

    # Disable internal blocking so Gemini can analyze potentially toxic inputs
    disable_safety = [
        types.SafetySetting(category=cat, threshold=types.HarmBlockThreshold.BLOCK_NONE)
        for cat in [
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
    ]

    # Use a simpler prompt and schema for Gemini
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",  # Ultra-fast model
        contents=f"Please moderate the following text:\n\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=GeminiSimpleModerationResult,  # Use the simpler schema here
            safety_settings=disable_safety,
            temperature=0.0,
            max_output_tokens=300,
        ),
    )

    # Extract parsed simple result from Gemini
    simple_result: GeminiSimpleModerationResult = response.parsed

    # Initialize OpenAI-compatible moderation result item
    item_result_content = ModerationResultItem(
        flagged=False,
        categories=Categories(),
        category_scores=CategoryScores(),
        category_applied_input_types=CategoryAppliedInputTypes()
    )

    if simple_result and simple_result.flagged:
        item_result_content.flagged = True
        # Map simple_result.primary_category to appropriate OpenAI categories and scores
        # For now, I'll put 'text' in sexual as a placeholder, this needs to be more granular if actual input types are desired
        item_result_content.category_applied_input_types.sexual.append("text")

        if simple_result.primary_category == HarmCategoryEnum.SEXUAL:
            item_result_content.categories.sexual = True
            item_result_content.category_scores.sexual = simple_result.confidence_score
        elif simple_result.primary_category == HarmCategoryEnum.HARASSMENT:
            item_result_content.categories.harassment = True
            item_result_content.category_scores.harassment = simple_result.confidence_score
        elif simple_result.primary_category == HarmCategoryEnum.VIOLENCE:
            item_result_content.categories.violence = True
            item_result_content.category_scores.violence = simple_result.confidence_score
        elif simple_result.primary_category == HarmCategoryEnum.TOXICITY:
            # Toxicity can map to hate or harassment in OpenAI's schema
            item_result_content.categories.hate = True
            item_result_content.category_scores.hate = simple_result.confidence_score
            item_result_content.categories.harassment = True
            item_result_content.category_scores.harassment = simple_result.confidence_score
        # PII and ERROR would not map to specific OpenAI categories here, but 'flagged' is already True.

    # If for some reason simple_result is None (e.g., severe API issue)
    if not simple_result:
        item_result_content.flagged = True  # Flag as error

    # Wrap inside OpenAI top-level metadata envelope
    full_response = OpenAIModerationResponse(
        id=f"modr-{uuid.uuid4().hex}",
        model="omni-moderation-latest",
        results=[item_result_content]
    )

    # Dump using aliases to guarantee slash/hyphen key names in JSON output
    return json.loads(full_response.model_dump_json(by_alias=True))


async def process_moderation_async(prompts: List[str]):
    all_results = []
    for prompt_text in prompts:
        try:
            # Call the synchronous moderation function
            result = moderate_openai_format(prompt_text)
            all_results.append(result)
        except Exception as e:
            # Create an error result in OpenAI format
            error_result_item = ModerationResultItem(
                flagged=True,
                categories=Categories(illicit=True),  # Indicate an error occurred by flagging a category
                category_scores=CategoryScores(illicit=1.0),  # High score for error
                category_applied_input_types=CategoryAppliedInputTypes(illicit=["api_error"])
            )
            error_response = OpenAIModerationResponse(
                id=f"modr-{uuid.uuid4().hex}-error",
                model="omni-moderation-latest-error",
                results=[error_result_item]
            )
            # Convert to dict to add custom error message and original prompt
            error_response_dict = json.loads(error_response.model_dump_json(by_alias=True))
            error_response_dict['error_message'] = str(e)
            error_response_dict['original_prompt'] = prompt_text  # Keep original prompt for context
            all_results.append(error_response_dict)
    return all_results


async def process_dataframe_moderation(df_part, output_file):
    # Extract all prompts from the 'prompt' column of the DataFrame
    df_prompts_to_moderate = [row['prompt'] for index, row in df_part.iterrows()]

    print("--- Gemini moderation processing started --")

    # Call the asynchronous moderation process
    df_moderation_results = await process_moderation_async(df_prompts_to_moderate)

    print("\n--- Gemini moderation processing complete --")

    # Iterate through the moderation results and the DataFrame rows simultaneously
    for i, result_obj in enumerate(df_moderation_results):
        # Get the corresponding row from the DataFrame
        # Using .iloc[i] to ensure correct matching based on order
        df_row = df_part.iloc[i]

        # Add 'prompt' and 'prompt_label' to the moderation result object
        result_obj['prompt'] = df_row['prompt']
        result_obj['prompt_label'] = df_row['prompt_label']

    # Write the results to the output JSONL file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for result in df_moderation_results:
            f.write(json.dumps(result) + '\n')

    print(f"Moderation results saved to: {output_file}")


def main(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Checking files in: {input_folder}\n")
    file_list = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])

    if not file_list:
        print(f"No JSON files found in the input directory: {input_folder}.")
        return
        
    processed = ["refusals_train.json_part_0001.json", "refusals_train.json_part_0002.json", "refusals_train.json_part_0003.json",
                "refusals_validation.json_part_0001.json",
                "test.json_part_0001.json", 
                "train.json_part_0001.json", "train.json_part_0002.json", "train.json_part_0003.json", "train.json_part_0004.json",
                "train.json_part_0005.json", "train.json_part_0006.json"
                ]

    for filename in file_list:
        input_file_path = os.path.join(input_folder, filename)
        output_file_name = f"{os.path.splitext(filename)[0]}-out.jsonl"
        output_file_path = os.path.join(output_folder, output_file_name)

        if any(sub in filename.lower() for sub in processed):
            print(f"Skipping ... file processed: {filename}\n")
            continue

        print(f"--- Reading file: {filename} ---")
        print(f"Output file: {output_file_path}\n")
        

        try:
            # Read the JSON file into a pandas DataFrame
            df_part = pd.read_json(input_file_path)
            print(f"Number of objects in {filename}: {len(df_part)}\n")

            asyncio.run(process_dataframe_moderation(df_part, output_file_path))

            print(f"File processing complete.")
        except Exception as e:
            print(f"Error processing {filename}: {e}\n")


if __name__ == "__main__":
    # Check API key first
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Set it in Windows CMD using: set GEMINI_API_KEY=YOUR_API_KEY")
        print("Or in PowerShell using: $env:GEMINI_API_KEY=\"YOUR_API_KEY\"")
        input("\nPress Enter to exit...")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Process JSON files for content moderation using Gemini API.")
    parser.add_argument('input_folder', type=str, nargs='?', default=None, help='Path to input folder.')
    parser.add_argument('output_folder', type=str, nargs='?', default=None, help='Path to output folder.')

    args = parser.parse_args()

    # Prompt interactively if missing command-line arguments
    input_folder = args.input_folder or input("Enter input folder path: ").strip()
    output_folder = args.output_folder or input("Enter output folder path: ").strip()

    try:
        main(input_folder, output_folder)
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

    input("\nPress Enter to exit...")
    
    