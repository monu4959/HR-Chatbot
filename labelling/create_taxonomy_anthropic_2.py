#!/usr/bin/env python3
"""
Create taxonomy from recognition data using iterative Claude analysis.

This is the first stage of the pipeline - creating a taxonomy from recognition data
that will be used in the annotation stage.

Modified to use Anthropic Direct API instead of AWS Bedrock.

Usage:
    python create_taxonomy.py
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
import json5
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime
import pickle
import copy
from tqdm import tqdm
import time
import google.generativeai as genai

# ============================================================================
# CONFIGURATION: Google AI Studio API Key (FREE)
# ============================================================================
# Read from the environment, never the code: GEMINI_API_KEY=... (or the first of GEMINI_API_KEYS)
GOOGLE_API_KEY = ([k.strip() for k in os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", "")).split(",") if k.strip()] or [''])[0]
# Get your key from: https://aistudio.google.com/
# ============================================================================


def calculate_cost(response: Dict[str, Any]) -> Tuple[int, int, float, float, float]:
    """
    Calculate the cost of a Claude API response.
    
    Args:
        response: The response from Claude
        
    Returns:
        Tuple of (input_tokens, output_tokens, input_cost, output_cost, total_cost)
    """
    # Constants for pricing (Anthropic Direct API)
    INPUT_COST_PER_MILLION = 3
    OUTPUT_COST_PER_MILLION = 15
    
    # Extract token counts
    input_tokens = response['usage']['input_tokens']
    output_tokens = response['usage']['output_tokens']
    
    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    
    total_cost = input_cost + output_cost
    
    return input_tokens, output_tokens, input_cost, output_cost, total_cost


def to_json(raw: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Convert raw string response to JSON, handling various formats.
    
    Args:
        raw: Raw string that might contain JSON
        verbose: Whether to print errors
        
    Returns:
        Parsed JSON as dict or None if parsing fails
    """
    if not raw:
        return None
    raw = raw.strip()
    map_not_found = lambda val, tgt: val if val >= 0 else tgt
    l_curly_idx = map_not_found(raw.find("{"), len(raw))
    r_curly_idx = map_not_found(raw.rfind("}"), 0) + 1
    l_square_idx = map_not_found(raw.find("["), len(raw))
    r_square_idx = map_not_found(raw.rfind("]"), 0) + 1

    start_idx = min(l_curly_idx, l_square_idx)
    end_idx = max(r_curly_idx, r_square_idx)

    if start_idx == len(raw) or end_idx == 1:
        return None
    try:
        return json5.loads(raw[start_idx:end_idx])
    except ValueError:
        return None


def get_claude_response(text: str, temperature: float = 1.0, max_tokens: int = 2000) -> Dict[str, Any]:
    """
    Get a response from Google Gemini (FREE tier).
    
    Args:
        text: The prompt text to send to the model
        temperature: Controls randomness in generation
        max_tokens: Maximum tokens in the response
        
    Returns:
        The response in a compatible format
    """
    # Configure the API
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Create the model
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    # Generation config
    generation_config = {
        'temperature': temperature,
        'max_output_tokens': max_tokens,
    }
    
    try:
        # Generate response
        response = model.generate_content(
            text,
            generation_config=generation_config
        )
        
        # Rate limiting - free tier is 15 requests/minute
        time.sleep(4)  # Wait 4 seconds between requests to be safe
        
        response_text = response.text
        
        # Convert to compatible format
        return {
            'content': [{'text': response_text}],
            'usage': {
                'input_tokens': 0,  # Gemini doesn't expose token counts easily
                'output_tokens': 0
            }
        }
    except Exception as e:
        print(f"Error calling Google Gemini: {e}")
        print("Waiting 10 seconds before retry...")
        time.sleep(10)
        # Retry once
        response = model.generate_content(text, generation_config=generation_config)
        return {
            'content': [{'text': response.text}],
            'usage': {'input_tokens': 0, 'output_tokens': 0}
        }


def get_content(response_body: Dict[str, Any]) -> str:
    """
    Extract the content from Claude's response.
    
    Args:
        response_body: The response body from Claude
        
    Returns:
        The text content
    """
    return response_body['content'][0]['text']


def format_recognition_info(row: pd.Series) -> str:
    """
    Format recognition information for a single row.
    
    Args:
        row: A pandas Series containing recognition data
        
    Returns:
        Formatted string with recognition information
    """
    parts = []
    
    # Nominator info (exclude name, only include available fields)
    nom_parts = []
    if pd.notna(row.get('nominator_title')):
        nom_parts.append(row['nominator_title'])
    
    if nom_parts:
        parts.append(f"Nominator: {' - '.join(nom_parts)}")
    
    # Recipient info (exclude name, only include available fields)
    rec_parts = []
    if pd.notna(row.get('recipient_title')):
        rec_parts.append(row['recipient_title'])
    
    if rec_parts:
        parts.append(f"Recipient: {' - '.join(rec_parts)}")
    
    # Recognition info
    rec_info_parts = []
    if pd.notna(row.get('award_title')):
        rec_info_parts.append(row['award_title'])
    
    if rec_info_parts:
        parts.append(f"Recognition: {' - '.join(rec_info_parts)}")
    
    # Message
    if pd.notna(row.get('message')):
        parts.append(f"Message: {row['message']}")
    
    return '\n'.join(parts)


def process_batch_with_pickle(
    batch_id: int, 
    df: pd.DataFrame, 
    existing_taxonomy: Union[Dict[str, Any], None] = None,
    responses_dir: str = "claude_responses"
) -> Dict[str, Any]:
    """
    Process a single batch of recognition data and save Claude's complete response as a pickle file.
    
    Args:
        batch_id: The batch ID to process
        df: DataFrame containing all data
        existing_taxonomy: The taxonomy from the previous batch, None for the first batch
        responses_dir: Directory to save Claude responses
        
    Returns:
        New additions to the taxonomy (JSON format, not including subcategory descriptions)
    """
    # Create directory for saving responses
    os.makedirs(responses_dir, exist_ok=True)
    
    # Get data for this batch
    batch_rows = df[df['batch_id'] == batch_id].copy()
    content = batch_rows['summary_dict_batch'].values
    
    if existing_taxonomy is None:
        # First batch - initialize taxonomy
        template = '''{
          "recognition_taxonomy": {
            "Category1": {
              "description":"description",
              "subcategories": [
                "Subcategory1","Subcategory2","Subcategory3",
                "..."
              ]
            },
            "..."
          }
        }'''
        
        prompt = f'''
        Role: You are a qualitative researcher. You use grounded theory to analyze the recongition data.
        <recognition data>
        {content}
        </>
        <Task> Create a high-level taxonomy about what are employee getting recognized for. 
        Include descriptions ONLY for main categories, not for subcategories.
        Concise description.
        Focus primarily on areas of impact instead of skills. 
        Use json output with following template (no descriptions for subcategories)
        {template}
        </>
        '''
        
        # Get Claude's response
        res = get_claude_response(prompt)
        
        # Save complete response as pickle file
        pickle_path = f"{responses_dir}/response_batch_{batch_id}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(res, f)
        
        content = get_content(res)
        content_json = to_json(content)
        
        # For the first batch, all content is new
        return content_json
    else:
        # Subsequent batches - only identify new additions to the existing taxonomy
        prompt = f'''
        Role: You are a qualitative researcher who uses grounded theory to analyze recognition data incrementally.
        <recognition_data>
        {content}
        </recognition_data>
        <existing_taxonomy>
        High-level taxonomy categorizing what employees are being recognized for:
        {existing_taxonomy}
        </existing_taxonomy>
        <task>
        1. Review the recognition data thoroughly.
        2. Categorize each recognition based on the existing taxonomy.
        3. Identify ONLY new categories or subcategories that do not exist in the current taxonomy.
        4. Return ONLY the new additions as a JSON object with the same structure as the existing taxonomy.
        5. Include descriptions ONLY for main categories, NOT for subcategories.
        6. If there are no new additions, return an empty taxonomy structure.
        7. Focus primarily on areas of impact instead of skills.
        8. DO NOT include any existing categories or subcategories in your output.
        </task>
        '''
        
        # Get Claude's response
        res = get_claude_response(prompt)
        
        # Save complete response as pickle file
        pickle_path = f"{responses_dir}/response_batch_{batch_id}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(res, f)
        
        content = get_content(res)
        additions = to_json(content)
        
        return additions


def merge_taxonomies(base_taxonomy: Dict[str, Any], additions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge additions into the base taxonomy (without subcategory descriptions).
    
    Args:
        base_taxonomy: Existing combined taxonomy
        additions: New additions to merge in
        
    Returns:
        Updated combined taxonomy
    """
    # Create a deep copy of the base taxonomy to avoid modifying the original
    result = copy.deepcopy(base_taxonomy)
    
    # Get the recognition_taxonomy dictionaries
    base_recognition = result.get("recognition_taxonomy", {})
    additions_recognition = additions.get("recognition_taxonomy", {})
    
    # Iterate through new categories
    for category_name, category_data in additions_recognition.items():
        if category_name in base_recognition:
            # Category exists, check subcategories
            base_subcategories = base_recognition[category_name].get("subcategories", [])
            new_subcategories = category_data.get("subcategories", [])
            
            # Add new subcategories that don't already exist
            for subcategory_name in new_subcategories:
                if subcategory_name not in base_subcategories:
                    base_subcategories.append(subcategory_name)
            
            # Update the subcategories in the base taxonomy
            base_recognition[category_name]["subcategories"] = base_subcategories
        else:
            # New category, add it with all its subcategories
            base_recognition[category_name] = category_data
    
    return result


def compress_taxonomy(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compress the taxonomy by merging overlapping or similar categories.
    
    Args:
        taxonomy: Current taxonomy to compress
        
    Returns:
        Compressed taxonomy with merged categories
    """
    prompt = f'''
    Role: You are a qualitative researcher specializing in taxonomy creation and optimization.
    <current_taxonomy>
    {taxonomy}
    </current_taxonomy>
    <task>
    1. Analyze this taxonomy and identify categories or subcategories that are conceptually similar or overlapping.
    2. Merge these similar items into more concise, representative categories.
    3. Ensure the compressed taxonomy maintains all important distinctions while reducing redundancy.
    4. Provide a cleaned, compressed taxonomy that captures the same information more efficiently.
    5. The compressed taxonomy should follow the same JSON structure as the input.
    6. Keep descriptions for main categories only, not for subcategories.
    </task>
    '''
    
    # Get response from Claude
    res = get_claude_response(prompt, max_tokens=5000)
    content = get_content(res)
    
    # Extract just the JSON part from the response
    compressed_json = extract_json_from_response(content)
    
    return compressed_json


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from Claude's response which may include explanations.
    
    Args:
        response: Full response from Claude
        
    Returns:
        Extracted JSON as dict
    """
    # Try to find JSON between triple backticks
    import re
    
    # Look for JSON pattern with either triple backticks or just the content
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, response)
    
    if matches:
        # Use the first match
        json_str = matches[0]
    else:
        # If no match with backticks, try to find a dictionary-like structure
        # This is a fallback in case Claude doesn't format with backticks
        json_str = response
    
    try:
        # Try to parse as JSON
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If direct parsing fails, use your to_json function
        return to_json(json_str)


def simplified_run_taxonomy_creation(
    df: pd.DataFrame, 
    start_batch: int = 1, 
    end_batch: int = 100,
    output_dir: str = "taxonomy_results",
    responses_dir: str = "claude_responses",
    compress_every: int = 10
) -> Dict[str, Any]:
    """
    Simplified taxonomy creation process, with periodic compression to reduce redundancy.
    
    Args:
        df: DataFrame containing all data
        start_batch: Starting batch ID (useful for resuming)
        end_batch: Final batch ID to process
        output_dir: Directory to save results
        responses_dir: Directory to save Claude responses
        compress_every: Compress taxonomy after this many batches
        
    Returns:
        Complete taxonomy built from all batches
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize merged taxonomy
    taxonomy = None
    
    # Check if resuming from a previous run
    if start_batch > 1:
        # Check if there's a compressed taxonomy
        previous_batch_range = ((start_batch - 1) // compress_every) * compress_every
        if previous_batch_range > 0:
            compressed_file = f"{output_dir}/taxonomy_batch_{previous_batch_range}.json"
            if os.path.exists(compressed_file):
                with open(compressed_file, 'r') as f:
                    taxonomy = json.load(f)
                    print(f"Loaded taxonomy from batch {previous_batch_range}")
    
    # Process batches
    for batch in range(start_batch, end_batch + 1):
        print(f"Processing batch {batch}...")
        
        try:
            # Process this batch to get new additions
            additions = process_batch_with_pickle(
                batch, df, taxonomy, responses_dir
            )
            
            # Update the taxonomy
            if taxonomy is None:
                # First batch, additions is the complete taxonomy
                taxonomy = additions
            else:
                # Subsequent batches, merge additions into the taxonomy
                taxonomy = merge_taxonomies(taxonomy, additions)
            
            # Save the updated taxonomy
            taxonomy_filename = f"{output_dir}/taxonomy_batch_{batch}.json"
            with open(taxonomy_filename, 'w') as f:
                json.dump(taxonomy, f, indent=2)
            
            print(f"- Taxonomy saved to {taxonomy_filename}")
            
            # Check if we need to compress the taxonomy
            if batch % compress_every == 0:
                print(f"Compressing taxonomy after batch {batch}...")
                compressed_taxonomy = compress_taxonomy(taxonomy)
                
                # Compressed taxonomy becomes the new main taxonomy
                taxonomy = compressed_taxonomy
                
                # Save the compressed taxonomy (overwrites current batch save)
                with open(taxonomy_filename, 'w') as f:
                    json.dump(taxonomy, f, indent=2)
                
                print(f"- Compressed taxonomy saved to {taxonomy_filename}")
            
        except Exception as e:
            print(f"Error processing batch {batch}: {str(e)}")
            # Save the last successful taxonomy for later resumption
            if taxonomy is not None:
                checkpoint_filename = f"{output_dir}/taxonomy_checkpoint_before_batch_{batch}.json"
                with open(checkpoint_filename, 'w') as f:
                    json.dump(taxonomy, f, indent=2)
                print(f"Saved checkpoint to {checkpoint_filename}")
            break
    
    return taxonomy


def add_ids_to_taxonomy(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add hierarchical IDs to taxonomy categories and subcategories.
    
    Args:
        taxonomy: Taxonomy without IDs
        
    Returns:
        Taxonomy with IDs added
    """
    # Define letters for main categories
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Copy the taxonomy to modify
    result = {"recognition_taxonomy": {}}
    
    # Iterate through main categories
    for i, (category_name, category_data) in enumerate(taxonomy["recognition_taxonomy"].items()):
        # Assign letter ID to category (A, B, C, etc.)
        category_id = letters[i]
        
        # Create new category entry with ID
        new_category_data = {
            "id": category_id,
            "description": category_data["description"]
        }
        
        # Process subcategories with numbered IDs
        subcategories_with_ids = []
        for j, subcategory in enumerate(category_data["subcategories"], 1):
            subcategory_id = f"{category_id}{j}"
            subcategories_with_ids.append({
                "id": subcategory_id,
                "name": subcategory
            })
        
        new_category_data["subcategories"] = subcategories_with_ids
        
        # Add to result
        result["recognition_taxonomy"][category_name] = new_category_data
    
    return result


def get_input_with_default(prompt: str, default: str, validation_fn=None) -> str:
    """
    Get user input with a default value and optional validation.
    
    Args:
        prompt: The prompt to show the user
        default: The default value
        validation_fn: Optional validation function
        
    Returns:
        The user input or default value
    """
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip() or default
        if validation_fn is None or validation_fn(user_input):
            return user_input
        print("Invalid input. Please try again.")


def main():
    """Main function to create taxonomy from recognition data."""
    # Check if API key is set
    if not GOOGLE_API_KEY:
        print("ERROR: set GEMINI_API_KEY (a Google AI Studio key from https://aistudio.google.com/) before running.")
        sys.exit(1)
    
    print("Taxonomy Creation for Employee Recognition (Using Google Gemini - FREE)")
    print("=" * 50)
    print("\nNote: Please ensure all data files are in the 'data/' directory")
    print("Note: Using Google AI Studio free tier (15 requests/minute)")
    print()
    
    # Get input file (single merged file)
    input_file = get_input_with_default(
        "Recognition data CSV filename",
        "mockup_awards.csv",
        lambda x: os.path.exists(f"data/{x}")
    )
    input_file = f"data/{input_file}"
    
    # Get sample size
    sample_size = int(get_input_with_default(
        "Sample size for taxonomy creation (recommended: 1000 for testing)",
        "1000",
        lambda x: x.isdigit() and int(x) > 0
    ))
    
    # Get batch processing parameters
    start_batch = int(get_input_with_default(
        "Start batch number (1 for new taxonomy)",
        "1",
        lambda x: x.isdigit() and int(x) > 0
    ))
    
    compress_every = int(get_input_with_default(
        "Compress taxonomy every N batches",
        "10",
        lambda x: x.isdigit() and int(x) > 0
    ))
    
    # Calculate default end batch based on sample size
    # 20 messages per batch, round down to nearest compression boundary
    total_batches = sample_size // 20
    default_end_batch = (total_batches // compress_every) * compress_every
    if default_end_batch == 0:
        default_end_batch = total_batches
    
    # Get end batch
    end_batch_input = get_input_with_default(
        f"End batch number (default: {default_end_batch} - latest compression batch)",
        str(default_end_batch),
        lambda x: x.isdigit() and int(x) >= start_batch
    )
    end_batch = int(end_batch_input)
    
    # Get output directory
    output_dir = get_input_with_default(
        "Output directory for taxonomy",
        "taxonomy_results"
    )
    
    responses_dir = get_input_with_default(
        "Directory for Claude responses",
        "claude_responses"
    )
    
    print("\nConfiguration Summary:")
    print(f"  Input file: {input_file}")
    print(f"  Sample size: {sample_size}")
    print(f"  Batch range: {start_batch} - {end_batch}")
    print(f"  Compress every: {compress_every} batches")
    print(f"  Output directory: {output_dir}")
    print(f"  Responses directory: {responses_dir}")
    
    confirm = input("\nProceed with these values? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Configuration cancelled.")
        sys.exit(0)
    
    print("\nLoading data...")
    
    # Load data
    df_final = pd.read_csv(input_file)
    
    print(f"Loaded {len(df_final)} records")
    print(f"Columns: {df_final.columns.tolist()}")
    
    # Sample data
    print(f"Sampling {sample_size} records...")
    df_sample = df_final.sample(n=min(sample_size, len(df_final)), random_state=42)
    
    # Drop duplicates by message
    print("Dropping duplicate messages...")
    df_unique = df_sample.drop_duplicates(subset=['message'])
    
    # Sort by message length (descending) - since we don't have 'value' column
    print("Sorting messages...")
    df_sorted = df_unique.copy()
    df_sorted['message_length'] = df_sorted['message'].str.len()
    df_sorted = df_sorted.sort_values('message_length', ascending=False).reset_index(drop=True)
    
    # Format recognition info for each row
    print("Formatting recognition data...")
    df_sorted['formatted_info'] = df_sorted.apply(format_recognition_info, axis=1)
    
    # Add batch information
    df_sorted['batch_id'] = np.ceil((np.arange(len(df_sorted)) + 1) / 20).astype(int)
    df_sorted['row_in_batch'] = 'row' + ((np.arange(len(df_sorted)) % 20) + 1).astype(str)
    df_sorted['summary_dict_batch'] = df_sorted['row_in_batch'] + ': ' + df_sorted['formatted_info'].astype(str)
    
    # Run taxonomy creation
    print(f"\nCreating taxonomy from batches {start_batch} to {end_batch}...")
    taxonomy = simplified_run_taxonomy_creation(
        df=df_sorted,
        start_batch=start_batch,
        end_batch=end_batch,
        output_dir=output_dir,
        responses_dir=responses_dir,
        compress_every=compress_every
    )
    
    if taxonomy:
        # Add IDs to the final taxonomy
        print("\nAdding IDs to taxonomy...")
        taxonomy_with_ids = add_ids_to_taxonomy(taxonomy)
        
        # Save the final taxonomy with IDs
        final_output = f"{output_dir}/compressed_taxonomy_with_ids.json"
        with open(final_output, 'w') as f:
            json.dump(taxonomy_with_ids, f, indent=2)
        
        print(f"\nSuccess! Final taxonomy saved to: {final_output}")
        
        # Print summary
        categories = taxonomy_with_ids['recognition_taxonomy']
        print(f"\nTaxonomy Summary:")
        print(f"  Total categories: {len(categories)}")
        for cat_name, cat_data in categories.items():
            print(f"  - {cat_name} ({cat_data['id']}): {len(cat_data['subcategories'])} subcategories")
    else:
        print("\nError: Taxonomy creation failed.")


if __name__ == '__main__':
    main()
