#!/usr/bin/env python3
"""
Recover taxonomy from incomplete pickle files.
Extracts categories and subcategories even from truncated JSON.
"""

import pickle
import json
import re
import os
from pathlib import Path

def extract_categories_from_incomplete_json(text):
    """Extract categories and subcategories from incomplete JSON using regex."""
    categories = {}
    
    # Pattern to find category blocks
    # Matches: "CategoryName": { "description": "...", "subcategories": [ "sub1", "sub2"
    category_pattern = r'"([^"]+)":\s*{\s*"description":\s*"([^"]+)",\s*"subcategories":\s*\[(.*?)(?:\]|$)'
    
    matches = re.findall(category_pattern, text, re.DOTALL)
    
    for cat_name, description, subcats_str in matches:
        # Extract subcategory names from the string
        subcat_pattern = r'"([^"]+)"'
        subcategories = re.findall(subcat_pattern, subcats_str)
        
        categories[cat_name] = {
            "description": description,
            "subcategories": subcategories
        }
    
    return categories

def merge_taxonomies(base, additions):
    """Merge new categories/subcategories into base."""
    if not base:
        return additions
    
    if not additions:
        return base
    
    # Merge each category
    for cat_name, cat_data in additions.items():
        if cat_name in base:
            # Category exists - merge subcategories
            existing_subs = set(base[cat_name].get("subcategories", []))
            new_subs = cat_data.get("subcategories", [])
            
            for sub in new_subs:
                if sub not in existing_subs:
                    base[cat_name]["subcategories"].append(sub)
        else:
            # New category
            base[cat_name] = cat_data
    
    return base

def recover_taxonomy():
    """Recover taxonomy from all pickle files."""
    pkl_dir = Path("claude_responses")
    
    if not pkl_dir.exists():
        print("❌ claude_responses directory not found!")
        return None
    
    pkl_files = sorted(pkl_dir.glob("response_batch_*.pkl"), 
                      key=lambda x: int(x.stem.split('_')[-1]))
    
    print(f"Found {len(pkl_files)} pickle files\n")
    
    all_categories = {}
    successful = 0
    failed = 0
    
    for pkl_file in pkl_files:
        batch_num = int(pkl_file.stem.split('_')[-1])
        
        try:
            # Load pickle
            with open(pkl_file, 'rb') as f:
                response = pickle.load(f)
            
            # Extract text
            text = response['content'][0]['text']
            
            # Extract categories (even from incomplete JSON)
            categories = extract_categories_from_incomplete_json(text)
            
            if categories:
                all_categories = merge_taxonomies(all_categories, categories)
                print(f"✅ Batch {batch_num:2d}: Found {len(categories)} categories")
                successful += 1
            else:
                print(f"⚠️  Batch {batch_num:2d}: No categories extracted")
                failed += 1
                
        except Exception as e:
            print(f"❌ Batch {batch_num:2d}: Error - {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Processed: {successful} successful, {failed} failed")
    print(f"Total unique categories found: {len(all_categories)}")
    
    if all_categories:
        # Create final taxonomy structure
        taxonomy = {"recognition_taxonomy": all_categories}
        
        # Save
        output_file = "taxonomy_results/recovered_taxonomy.json"
        with open(output_file, 'w') as f:
            json.dump(taxonomy, f, indent=2)
        
        print(f"\n✅ Saved to: {output_file}")
        
        # Summary
        print("\n📊 Category Summary:")
        print("="*60)
        for cat_name, cat_data in sorted(all_categories.items()):
            num_subs = len(cat_data.get('subcategories', []))
            print(f"  {cat_name}")
            print(f"    → {num_subs} subcategories")
            if num_subs <= 5:
                for sub in cat_data.get('subcategories', []):
                    print(f"       - {sub}")
        
        return taxonomy
    else:
        print("\n❌ No categories recovered")
        return None

if __name__ == '__main__':
    print("="*60)
    print("Taxonomy Recovery Script (Handles Incomplete JSON)")
    print("="*60)
    print()
    recover_taxonomy()