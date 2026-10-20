#!/usr/bin/env python3
"""
Create taxonomy from recognition data - V2

Changes from FIXED VERSION:
1. Sleep increased from 4s to 8s to stay under 10 req/min free tier limit
2. Multiple API keys with automatic rotation on 429 rate limit errors
3. 429-specific retry: waits 60s then switches to next API key
4. All outputs saved to _1 directories to avoid overwriting previous results
5. genai.configure() moved outside request loop (called once per key switch)
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
# Keys come from the environment, never the code: GEMINI_API_KEYS="key1,key2,..." (rotated on 429s)
GOOGLE_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", "")).split(",") if k.strip()]

# Track which key is currently active
current_key_index = 0

# ============================================================================


def get_active_api_key() -> str:
    """Return the currently active API key."""
    return GOOGLE_API_KEYS[current_key_index]


def rotate_api_key() -> bool:
    """
    Switch to the next available API key.
    Returns True if successfully rotated, False if no more keys available.
    """
    global current_key_index
    if current_key_index + 1 < len(GOOGLE_API_KEYS):
        current_key_index += 1
        new_key = GOOGLE_API_KEYS[current_key_index]
        genai.configure(api_key=new_key)
        print(f"🔄 Switched to API key #{current_key_index + 1}")
        return True
    else:
        print("❌ No more API keys available to rotate to!")
        return False


def initialize_genai():
    """Initialize genai with the current active key. Call once at startup."""
    if not GOOGLE_API_KEYS:
        sys.exit("Set GEMINI_API_KEYS (comma-separated Google AI Studio keys) before running.")
    genai.configure(api_key=get_active_api_key())
    print(f"✅ Initialized with API key #{current_key_index + 1}")


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
    Sends request to Gemini with the following retry strategy on 429:

    Per-key attempts: Try each key TWICE before moving on
      - Key 1, attempt 1 → wait 60s → Key 1, attempt 2 → wait 60s → move to Key 2
      - Key 2, attempt 1 → wait 60s → Key 2, attempt 2 → wait 60s → move to Key 3
      - Key 3, attempt 1 → wait 60s → Key 3, attempt 2 → wait 60s → all keys exhausted

    Full cycle retry: If ALL keys fail, do one full second pass through all keys again
    before finally giving up and reporting which batch it stopped at.

    Sleep: 15s between successful calls (safe buffer under 10 req/min limit)
    Non-429 errors: standard 10s/20s/30s backoff
    """
    global current_key_index

    generation_config = {
        'temperature': temperature,
        'max_output_tokens': max_tokens,
    }

    def try_with_current_key(attempt_label: str):
        """Try one request with whatever key is currently active."""
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(text, generation_config=generation_config)
        time.sleep(15)  # 15s sleep — safe buffer under 10 req/min free tier limit
        response_text = response.text
        if not response_text or len(response_text) < 50:
            raise ValueError("Response too short or empty")
        return {
            'content': [{'text': response_text}],
            'usage': {'input_tokens': 0, 'output_tokens': 0}
        }

    def is_rate_limit_error(e: Exception) -> bool:
        err = str(e)
        return "429" in err or "quota" in err.lower() or "rate" in err.lower()

    def attempt_all_keys(pass_label: str):
        """
        Try every key twice. Returns response if any key succeeds.
        Raises RuntimeError if all keys are exhausted.
        """
        global current_key_index
        num_keys = len(GOOGLE_API_KEYS)

        for key_idx in range(num_keys):
            current_key_index = key_idx
            genai.configure(api_key=GOOGLE_API_KEYS[current_key_index])

            for per_key_attempt in range(1, 3):  # 2 attempts per key
                label = f"{pass_label} | Key {key_idx + 1}/{num_keys}, attempt {per_key_attempt}/2"
                try:
                    print(f"   🔑 Trying: {label}")
                    return try_with_current_key(label)
                except Exception as e:
                    if is_rate_limit_error(e):
                        print(f"   ⚠️  Rate limit on {label}. Waiting 60s...")
                        time.sleep(60)
                        # After 2nd attempt on this key, move to next key
                        if per_key_attempt == 2:
                            print(f"   ➡️  Key {key_idx + 1} exhausted, moving to next key...")
                    else:
                        # Non-rate-limit: don't burn all keys, just backoff and retry
                        wait = 10 * per_key_attempt
                        print(f"   ⚠️  Non-rate-limit error on {label}: {e}. Waiting {wait}s...")
                        time.sleep(wait)

        raise RuntimeError("All API keys exhausted in this pass.")

    # --- Main logic: two full passes through all keys ---
    for full_pass in range(1, 3):  # Pass 1, then Pass 2 if needed
        try:
            return attempt_all_keys(f"Pass {full_pass}")
        except RuntimeError:
            if full_pass == 1:
                print(f"\n🔄 All keys failed on Pass 1. Starting Pass 2 through all keys...")
                time.sleep(30)  # Brief pause before second full cycle
            else:
                # Both passes exhausted — caller will handle the batch number reporting
                raise RuntimeError("All API keys failed across 2 full passes. Stopping.")


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
    responses_dir: str = "claude_responses_1"   # FIX: _1 suffix
) -> Dict[str, Any]:
    """
    Process a single batch and save response pickle.
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
        
        Output ONLY valid JSON. No explanation, no markdown, no preamble.
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
        
        Output ONLY valid JSON. No explanation, no markdown, no preamble.
        IMPORTANT: Provide COMPLETE, VALID JSON. Close all brackets properly.
        </task>
        '''
    
    # Get response with retries
    res = get_claude_response(prompt)
    
    # Save response pickle
    pickle_path = f"{responses_dir}/response_batch_{batch_id}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(res, f)
    
    # Parse content
    content = get_content(res)
    content_json = to_json_improved(content)
    
    if content_json is None:
        print(f"⚠️  WARNING: Batch {batch_id} returned invalid JSON")
        print(f"   Response preview: {content[:200]}...")
        return {"recognition_taxonomy": {}}
    
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
    Compress taxonomy by merging similar/overlapping categories.
    Uses higher token limit since output is the full taxonomy.
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
    6. Aim for 8-15 main categories total. Merge any categories with fewer than 3 subcategories into related ones.
    
    Output ONLY valid JSON. No explanation, no markdown, no preamble.
    IMPORTANT: Provide COMPLETE, VALID JSON. Close all brackets properly.
    </task>
    '''
    
    res = get_claude_response(prompt, max_tokens=12000)
    content = get_content(res)
    compressed_json = to_json_improved(content)
    
    if compressed_json is None:
        print("⚠️  Compression failed, returning original taxonomy")
        return taxonomy
    
    return compressed_json


def simplified_run_taxonomy_creation(
    df: pd.DataFrame, 
    start_batch: int = 1, 
    end_batch: int = 100,
    output_dir: str = "taxonomy_results_1",       # FIX: _1 suffix
    responses_dir: str = "claude_responses_1",    # FIX: _1 suffix
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
        print(f"Processing batch {batch}/{end_batch}...")
        
        try:
            additions = process_batch_with_pickle(batch, df, taxonomy, responses_dir)
            
            if taxonomy is None:
                taxonomy = additions
            else:
                taxonomy = merge_taxonomies(taxonomy, additions)
            
            if taxonomy and "recognition_taxonomy" in taxonomy:
                taxonomy_filename = f"{output_dir}/taxonomy_batch_{batch}.json"
                with open(taxonomy_filename, 'w') as f:
                    json.dump(taxonomy, f, indent=2)
                print(f"✅ Saved: {taxonomy_filename}")
            else:
                print(f"⚠️  Batch {batch} - taxonomy is invalid, skipping save")
            
            if batch % compress_every == 0:
                print(f"🗜️  Compressing taxonomy after batch {batch}...")
                compressed_taxonomy = compress_taxonomy(taxonomy)
                
                if compressed_taxonomy and "recognition_taxonomy" in compressed_taxonomy:
                    taxonomy = compressed_taxonomy
                    with open(taxonomy_filename, 'w') as f:
                        json.dump(taxonomy, f, indent=2)
                    print(f"✅ Compressed taxonomy saved")
                else:
                    print(f"⚠️  Compression failed, keeping original")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ PIPELINE STOPPED at batch {batch}/{end_batch}")
            print(f"   Reason: {str(e)}")
            print(f"   Batches completed: {batch - start_batch} of {end_batch - start_batch + 1}")
            print(f"   To resume, set start_batch = {batch} when prompted")
            print(f"{'='*60}\n")
            if taxonomy is not None:
                checkpoint_filename = f"{output_dir}/taxonomy_checkpoint_before_batch_{batch}.json"
                with open(checkpoint_filename, 'w') as f:
                    json.dump(taxonomy, f, indent=2)
                print(f"💾 Progress saved to: {checkpoint_filename}")
                print(f"   Resume tip: run script again and enter start_batch={batch}")
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
    # Validate API keys
    valid_keys = [k for k in GOOGLE_API_KEYS if k and not k.startswith("YOUR_API_KEY")]
    if not valid_keys:
        print("ERROR: Please set at least one valid Google AI Studio API key in GOOGLE_API_KEYS!")
        sys.exit(1)
    
    # FIX: Initialize genai ONCE here, not inside every request
    initialize_genai()
    
    print("\nTaxonomy Creation - V2")
    print("="*50)
    print(f"API keys loaded: {len(valid_keys)} key(s) available")
    print("Rate limit fix: 15s sleep between calls (was 4s)")
    print("Output dirs: taxonomy_results_1 / claude_responses_1")
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
    
    # FIX: Hardcoded _1 output directories
    output_dir = "taxonomy_results_1"
    responses_dir = "claude_responses_1"
    
    print("\nConfiguration:")
    print(f"  Input:        {input_file}")
    print(f"  Sample size:  {sample_size}")
    print(f"  Batches:      {start_batch} → {end_batch}")
    print(f"  Compress every: {compress_every} batches")
    print(f"  Output dir:   {output_dir}")
    print(f"  Responses dir:{responses_dir}")
    print(f"  Sleep/call:   15s (extra safe buffer for 10 req/min limit)")
    print(f"  API keys:     {len(valid_keys)} available (auto-rotates on 429)")
    
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
    
    print(f"  Records loaded: {len(df_final)}")
    print(f"  After dedup:    {len(df_unique)}")
    print(f"  Total batches:  {df_sorted['batch_id'].max()}")
    
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
        
        # FIX: Final output also goes to _1 directory
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