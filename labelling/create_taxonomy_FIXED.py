#!/usr/bin/env python3
"""
Create taxonomy from recognition data - FIXED VERSION

Key improvements:
1. Increased max_tokens from 2000 to 8000
2. Better error handling and retries
3. Validates responses before saving
4. Improved JSON parsing with fallback methods
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
import re
import google.generativeai as genai

# ============================================================================
# CONFIGURATION
# ============================================================================
# Read from the environment, never the code: GEMINI_API_KEY=... (or the first of GEMINI_API_KEYS)
GOOGLE_API_KEY = ([k.strip() for k in os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", "")).split(",") if k.strip()] or [''])[0]
# ============================================================================


def to_json_improved(raw: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Improved JSON parsing with multiple fallback strategies.
    """
    if not raw:
        return None
    
    raw = raw.strip()
    
    # Strategy 1: Try to extract from markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, raw)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except:
                pass
    
    # Strategy 2: Find JSON structure boundaries
    map_not_found = lambda val, tgt: val if val >= 0 else tgt
    l_curly_idx = map_not_found(raw.find("{"), len(raw))
    r_curly_idx = map_not_found(raw.rfind("}"), 0) + 1
    l_square_idx = map_not_found(raw.find("["), len(raw))
    r_square_idx = map_not_found(raw.rfind("]"), 0) + 1

    start_idx = min(l_curly_idx, l_square_idx)
    end_idx = max(r_curly_idx, r_square_idx)

    if start_idx < len(raw) and end_idx > 1:
        try:
            return json5.loads(raw[start_idx:end_idx])
        except:
            pass
    
    # Strategy 3: Try regex extraction for incomplete JSON
    try:
        categories = {}
        category_pattern = r'"([^"]+)":\s*{\s*"description":\s*"([^"]+)",\s*"subcategories":\s*\[(.*?)(?:\]|$)'
        matches = re.findall(category_pattern, raw, re.DOTALL)
        
        for cat_name, description, subcats_str in matches:
            subcat_pattern = r'"([^"]+)"'
            subcategories = re.findall(subcat_pattern, subcats_str)
            categories[cat_name] = {
                "description": description,
                "subcategories": subcategories
            }
        
        if categories:
            return {"recognition_taxonomy": categories}
    except:
        pass
    
    if verbose:
        print("⚠️  All JSON parsing strategies failed")
    
    return None


def get_claude_response(text: str, temperature: float = 1.0, max_tokens: int = 8000) -> Dict[str, Any]:
    """
    FIXED: Increased max_tokens and added retries.
    """
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    generation_config = {
        'temperature': temperature,
        'max_output_tokens': max_tokens,  # FIXED: Was 2000, now 8000
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(text, generation_config=generation_config)
            time.sleep(4)  # Rate limiting
            
            response_text = response.text
            
            # FIXED: Validate response
            if not response_text or len(response_text) < 50:
                raise ValueError("Response too short or empty")
            
            return {
                'content': [{'text': response_text}],
                'usage': {'input_tokens': 0, 'output_tokens': 0}
            }
            
        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise


def get_content(response_body: Dict[str, Any]) -> str:
    return response_body['content'][0]['text']


def format_recognition_info(row: pd.Series) -> str:
    parts = []
    
    nom_parts = []
    if pd.notna(row.get('nominator_title')):
        nom_parts.append(row['nominator_title'])
    if nom_parts:
        parts.append(f"Nominator: {' - '.join(nom_parts)}")
    
    rec_parts = []
    if pd.notna(row.get('recipient_title')):
        rec_parts.append(row['recipient_title'])
    if rec_parts:
        parts.append(f"Recipient: {' - '.join(rec_parts)}")
    
    rec_info_parts = []
    if pd.notna(row.get('award_title')):
        rec_info_parts.append(row['award_title'])
    if rec_info_parts:
        parts.append(f"Recognition: {' - '.join(rec_info_parts)}")
    
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
    FIXED: Better error handling and validation.
    """
    os.makedirs(responses_dir, exist_ok=True)
    
    batch_rows = df[df['batch_id'] == batch_id].copy()
    content = batch_rows['summary_dict_batch'].values
    
    if existing_taxonomy is None:
        template = '''{
          "recognition_taxonomy": {
            "Category1": {
              "description":"description",
              "subcategories": ["Subcategory1","Subcategory2","Subcategory3"]
            }
          }
        }'''
        
        prompt = f'''
        Role: You are a qualitative researcher using grounded theory.
        <recognition data>
        {content}
        </>
        <Task>
        Create a high-level taxonomy about what employees are getting recognized for.
        Include descriptions ONLY for main categories, not for subcategories.
        Focus primarily on areas of impact instead of skills.
        Use json output with following template:
        {template}
        
        IMPORTANT: Provide COMPLETE, VALID JSON. Close all brackets properly.
        </>
        '''
    else:
        prompt = f'''
        Role: You are a qualitative researcher using grounded theory.
        <recognition_data>
        {content}
        </recognition_data>
        <existing_taxonomy>
        {existing_taxonomy}
        </existing_taxonomy>
        <task>
        1. Review the recognition data thoroughly.
        2. Categorize each recognition based on the existing taxonomy.
        3. Identify ONLY new categories or subcategories that do not exist.
        4. Return ONLY the new additions as JSON.
        5. If there are no new additions, return empty taxonomy: {{"recognition_taxonomy": {{}}}}
        6. Focus on areas of impact instead of skills.
        7. DO NOT include any existing categories or subcategories.
        
        IMPORTANT: Provide COMPLETE, VALID JSON. Close all brackets properly.
        </task>
        '''
    
    # Get response with retries
    res = get_claude_response(prompt)
    
    # Save response
    pickle_path = f"{responses_dir}/response_batch_{batch_id}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(res, f)
    
    # Parse content with improved function
    content = get_content(res)
    content_json = to_json_improved(content)  # FIXED: Using improved parser
    
    # FIXED: Validate before returning
    if content_json is None:
        print(f"⚠️  WARNING: Batch {batch_id} returned invalid JSON")
        print(f"   Response preview: {content[:200]}...")
        return {"recognition_taxonomy": {}}  # Return empty instead of None
    
    return content_json


def merge_taxonomies(base_taxonomy: Dict[str, Any], additions: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base_taxonomy)
    
    base_recognition = result.get("recognition_taxonomy", {})
    additions_recognition = additions.get("recognition_taxonomy", {})
    
    for category_name, category_data in additions_recognition.items():
        if category_name in base_recognition:
            base_subcategories = base_recognition[category_name].get("subcategories", [])
            new_subcategories = category_data.get("subcategories", [])
            
            for subcategory_name in new_subcategories:
                if subcategory_name not in base_subcategories:
                    base_subcategories.append(subcategory_name)
            
            base_recognition[category_name]["subcategories"] = base_subcategories
        else:
            base_recognition[category_name] = category_data
    
    return result


def compress_taxonomy(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Increased max_tokens for compression.
    """
    prompt = f'''
    Role: You are a qualitative researcher specializing in taxonomy optimization.
    <current_taxonomy>
    {json.dumps(taxonomy, indent=2)}
    </current_taxonomy>
    <task>
    1. Analyze this taxonomy and identify similar or overlapping categories/subcategories.
    2. Merge similar items into more concise, representative categories.
    3. Maintain all important distinctions while reducing redundancy.
    4. Provide a cleaned, compressed taxonomy in the same JSON format.
    5. Keep descriptions for main categories only, not for subcategories.
    
    IMPORTANT: Provide COMPLETE, VALID JSON. Close all brackets properly.
    </task>
    '''
    
    res = get_claude_response(prompt, max_tokens=12000)  # FIXED: Even higher for compression
    content = get_content(res)
    compressed_json = to_json_improved(content)  # FIXED: Using improved parser
    
    if compressed_json is None:
        print("⚠️  Compression failed, returning original taxonomy")
        return taxonomy
    
    return compressed_json


def simplified_run_taxonomy_creation(
    df: pd.DataFrame, 
    start_batch: int = 1, 
    end_batch: int = 100,
    output_dir: str = "taxonomy_results",
    responses_dir: str = "claude_responses",
    compress_every: int = 10
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    taxonomy = None
    
    if start_batch > 1:
        previous_batch_range = ((start_batch - 1) // compress_every) * compress_every
        if previous_batch_range > 0:
            compressed_file = f"{output_dir}/taxonomy_batch_{previous_batch_range}.json"
            if os.path.exists(compressed_file):
                with open(compressed_file, 'r') as f:
                    taxonomy = json.load(f)
                    print(f"Loaded taxonomy from batch {previous_batch_range}")
    
    for batch in range(start_batch, end_batch + 1):
        print(f"Processing batch {batch}...")
        
        try:
            additions = process_batch_with_pickle(batch, df, taxonomy, responses_dir)
            
            if taxonomy is None:
                taxonomy = additions
            else:
                taxonomy = merge_taxonomies(taxonomy, additions)
            
            # FIXED: Validate before saving
            if taxonomy and "recognition_taxonomy" in taxonomy:
                taxonomy_filename = f"{output_dir}/taxonomy_batch_{batch}.json"
                with open(taxonomy_filename, 'w') as f:
                    json.dump(taxonomy, f, indent=2)
                print(f"✅ Taxonomy saved to {taxonomy_filename}")
            else:
                print(f"⚠️  Batch {batch} - taxonomy is invalid, skipping save")
            
            if batch % compress_every == 0:
                print(f"Compressing taxonomy after batch {batch}...")
                compressed_taxonomy = compress_taxonomy(taxonomy)
                
                if compressed_taxonomy and "recognition_taxonomy" in compressed_taxonomy:
                    taxonomy = compressed_taxonomy
                    with open(taxonomy_filename, 'w') as f:
                        json.dump(taxonomy, f, indent=2)
                    print(f"✅ Compressed taxonomy saved")
                else:
                    print(f"⚠️  Compression failed, keeping original")
            
        except Exception as e:
            print(f"❌ Error processing batch {batch}: {str(e)}")
            if taxonomy is not None:
                checkpoint_filename = f"{output_dir}/taxonomy_checkpoint_before_batch_{batch}.json"
                with open(checkpoint_filename, 'w') as f:
                    json.dump(taxonomy, f, indent=2)
                print(f"Saved checkpoint to {checkpoint_filename}")
            break
    
    return taxonomy


def add_ids_to_taxonomy(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = {"recognition_taxonomy": {}}
    
    for i, (category_name, category_data) in enumerate(taxonomy["recognition_taxonomy"].items()):
        if i >= len(letters):
            print(f"⚠️  Warning: More than 26 categories, some won't have letter IDs")
            break
            
        category_id = letters[i]
        
        new_category_data = {
            "id": category_id,
            "description": category_data.get("description", "")
        }
        
        subcategories_with_ids = []
        for j, subcategory in enumerate(category_data.get("subcategories", []), 1):
            subcategory_id = f"{category_id}{j}"
            subcategories_with_ids.append({
                "id": subcategory_id,
                "name": subcategory
            })
        
        new_category_data["subcategories"] = subcategories_with_ids
        result["recognition_taxonomy"][category_name] = new_category_data
    
    return result


def get_input_with_default(prompt: str, default: str, validation_fn=None) -> str:
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip() or default
        if validation_fn is None or validation_fn(user_input):
            return user_input
        print("Invalid input. Please try again.")


def main():
    if not GOOGLE_API_KEY:
        print("ERROR: set GEMINI_API_KEY (your Google AI Studio API key) before running.")
        sys.exit(1)
    
    print("Taxonomy Creation - FIXED VERSION")
    print("="*50)
    print("Fixes: Increased token limits, better error handling, improved JSON parsing")
    print()
    
    input_file = get_input_with_default(
        "Recognition data CSV filename",
        "mockup_awards.csv",
        lambda x: os.path.exists(f"data/{x}")
    )
    input_file = f"data/{input_file}"
    
    sample_size = int(get_input_with_default(
        "Sample size for taxonomy creation",
        "1000",
        lambda x: x.isdigit() and int(x) > 0
    ))
    
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
    
    total_batches = sample_size // 20
    default_end_batch = (total_batches // compress_every) * compress_every
    if default_end_batch == 0:
        default_end_batch = total_batches
    
    end_batch = int(get_input_with_default(
        f"End batch number (default: {default_end_batch})",
        str(default_end_batch),
        lambda x: x.isdigit() and int(x) >= start_batch
    ))
    
    output_dir = get_input_with_default("Output directory", "taxonomy_results")
    responses_dir = get_input_with_default("Responses directory", "claude_responses")
    
    print("\nConfiguration:")
    print(f"  Input: {input_file}")
    print(f"  Sample: {sample_size}")
    print(f"  Batches: {start_batch} - {end_batch}")
    print(f"  Max tokens: 8000 (was 2000)")
    
    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm == 'n':
        sys.exit(0)
    
    print("\nLoading data...")
    df_final = pd.read_csv(input_file)
    df_sample = df_final.sample(n=min(sample_size, len(df_final)), random_state=42)
    df_unique = df_sample.drop_duplicates(subset=['message'])
    
    df_sorted = df_unique.copy()
    df_sorted['message_length'] = df_sorted['message'].str.len()
    df_sorted = df_sorted.sort_values('message_length', ascending=False).reset_index(drop=True)
    
    df_sorted['formatted_info'] = df_sorted.apply(format_recognition_info, axis=1)
    df_sorted['batch_id'] = np.ceil((np.arange(len(df_sorted)) + 1) / 20).astype(int)
    df_sorted['row_in_batch'] = 'row' + ((np.arange(len(df_sorted)) % 20) + 1).astype(str)
    df_sorted['summary_dict_batch'] = df_sorted['row_in_batch'] + ': ' + df_sorted['formatted_info'].astype(str)
    
    print(f"\nCreating taxonomy from batches {start_batch} to {end_batch}...")
    taxonomy = simplified_run_taxonomy_creation(
        df=df_sorted,
        start_batch=start_batch,
        end_batch=end_batch,
        output_dir=output_dir,
        responses_dir=responses_dir,
        compress_every=compress_every
    )
    
    if taxonomy and "recognition_taxonomy" in taxonomy:
        print("\nAdding IDs to taxonomy...")
        taxonomy_with_ids = add_ids_to_taxonomy(taxonomy)
        
        final_output = f"{output_dir}/compressed_taxonomy_with_ids.json"
        with open(final_output, 'w') as f:
            json.dump(taxonomy_with_ids, f, indent=2)
        
        print(f"\n✅ Success! Final taxonomy saved to: {final_output}")
        
        categories = taxonomy_with_ids['recognition_taxonomy']
        print(f"\n📊 Taxonomy Summary:")
        print(f"  Total categories: {len(categories)}")
        for cat_name, cat_data in categories.items():
            print(f"  - {cat_name} ({cat_data['id']}): {len(cat_data['subcategories'])} subcategories")
    else:
        print("\n❌ Error: Taxonomy creation failed.")


if __name__ == '__main__':
    main()
