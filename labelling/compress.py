#!/usr/bin/env python3
"""
Intelligent taxonomy compression - merges similar categories and subcategories.
"""

import json
from collections import defaultdict

def merge_similar_categories(taxonomy):
    """
    Merge similar categories based on semantic similarity.
    """
    categories = taxonomy['recognition_taxonomy']
    
    # Define merge groups - categories that should be combined
    merge_groups = {
        "Collaboration & Teamwork": [
            "Collaboration & Partnership",
            "Collaboration & Teamwork",
            "Collaboration and Support",
            "Collaborative Support & Partnership",
            "Team & Cross-Functional Collaboration"
        ],
        "Operational Excellence": [
            "Operational Excellence",
            "Operational Excellence & Delivery",
            "Operational Excellence & Efficiency",
            "Operational Excellence & Impact",
            "Operational Excellence & Innovation",
            "Operational Excellence & Problem Resolution",
            "Operational Excellence & Project Achievement",
            "Operational Excellence & Project Delivery",
            "Operational Excellence & Project Execution",
            "Operational Excellence & Project Success",
            "Operational Excellence & Support",
            "Operational Efficiency & Problem Resolution",
            "Driving Operational Excellence",
            "Driving Operational Excellence & Efficiency",
            "Operational Support & Problem Solving"
        ],
        "Project & Product Delivery": [
            "Project & Product Delivery",
            "Product & Solution Delivery",
            "Impactful Project & Task Execution",
            "Exceptional Contribution to Projects & Deliverables",
            "Project and Initiative Advancement",
            "Work & Project Contribution"
        ],
        "Strategic Business Impact": [
            "Strategic Business Impact",
            "Strategic Impact & Innovation",
            "Driving Strategic Outcomes",
            "Driving Business Impact",
            "Driving Business Outcomes",
            "Driving Business & Project Outcomes",
            "Driving Business & Operational Outcomes",
            "Driving Business & Strategic Impact"
        ],
        "Customer & Client Success": [
            "Customer & Client Success"
        ],
        "Responsive Support": [
            "Timely & Responsive Support"
        ]
    }
    
    # Create new taxonomy
    merged_taxonomy = {}
    
    for target_name, source_names in merge_groups.items():
        # Collect all subcategories and descriptions from source categories
        all_subcategories = []
        descriptions = []
        
        for source_name in source_names:
            if source_name in categories:
                cat_data = categories[source_name]
                
                # Collect description
                if 'description' in cat_data and cat_data['description']:
                    descriptions.append(cat_data['description'])
                
                # Collect subcategories
                subcats = cat_data.get('subcategories', [])
                all_subcategories.extend(subcats)
        
        if all_subcategories or descriptions:
            # Use the longest/most detailed description
            best_description = max(descriptions, key=len) if descriptions else ""
            
            # Deduplicate and clean subcategories
            unique_subcats = deduplicate_subcategories(all_subcategories)
            
            merged_taxonomy[target_name] = {
                "description": best_description,
                "subcategories": unique_subcats
            }
    
    return {"recognition_taxonomy": merged_taxonomy}

def deduplicate_subcategories(subcategories):
    """
    Remove duplicate subcategories and merge very similar ones.
    """
    # Merge rules for subcategories
    merge_map = {
        # Process-related
        "Process Improvement": ["Process Improvement", "Process Improvement & Optimization", 
                                "Process Improvement & Simplification", "Process Efficiency"],
        "System Development & Support": ["System Development & Support", "System Enhancement & Maintenance",
                                         "System Maintenance & Troubleshooting"],
        
        # Project-related
        "Project Delivery & Execution": ["Project Delivery", "Project Delivery & Execution", 
                                         "Project Completion & Delivery", "Timely Task Completion",
                                         "Project Management & Coordination", "Project Delivery & Milestones"],
        "Quality Deliverables": ["Quality Deliverables", "Quality and Detail Orientation"],
        
        # Collaboration
        "Cross-functional Collaboration": ["Cross-functional Collaboration", "Cross-functional Partnership",
                                           "Cross-functional Assistance"],
        "Teamwork & Mutual Support": ["Teamwork and Mutual Support", "Teamwork & Cooperation",
                                      "Teamwork and Mutual Aid", "Teamwork & Mutual Aid",
                                      "Mutual Support & Assistance", "Mutual Support & Coverage"],
        "Knowledge Sharing & Mentorship": ["Mentorship and Knowledge Sharing", "Knowledge Sharing & Guidance"],
        
        # Problem-solving
        "Problem Solving & Resolution": ["Problem Solving & Innovation", "Problem Resolution & Troubleshooting",
                                         "Problem Solving & Issue Resolution", "Issue Resolution & Problem Solving",
                                         "Technical Troubleshooting", "Issue Resolution"],
        
        # Innovation & Strategy
        "Innovation & Strategic Initiatives": ["Technical Innovation & Problem Solving", "Strategic Project Delivery",
                                               "Strategic Product Initiatives", "Strategic Planning & Alignment"],
        
        # Business impact
        "Business Growth & Development": ["Business Development & Growth", "Business Growth & Visibility",
                                          "Business Growth & Expansion", "Market & Customer Acquisition"],
        "Revenue & Pipeline Impact": ["Influencing Revenue & Pipeline Growth"],
        "Goal Achievement": ["Achieving Project Milestones & Deliverables", "Project Success & Goal Achievement",
                            "Contribution to Strategic Goals"],
        
        # Customer-related
        "Customer Service Excellence": ["Exceptional Service Delivery"],
        "Client Relationship Management": ["Client Relationship Management"],
        
        # Support & responsiveness  
        "Rapid Response & Support": ["Speedy Resolution", "Prompt Action", "Urgent Assistance"],
        
        # Operations
        "Efficiency & Productivity": ["Efficiency & Productivity"],
        "Operational Support": ["Administrative Support", "Covering tasks and projects"],
        
        # Product/Feature work
        "Feature Development": ["Feature Development & Design", "Successful Launch & Implementation"],
        
        # Other
        "Automation": ["Process Improvement & Automation"],
        "Accessibility": ["Accessibility Support"],
        "System Integration": ["System Integration/Sync"],
        "Vulnerability Management": ["Vulnerability Reduction"]
    }
    
    # Create reverse lookup
    subcat_to_canonical = {}
    for canonical, variants in merge_map.items():
        for variant in variants:
            subcat_to_canonical[variant] = canonical
    
    # Map and deduplicate
    canonical_subcats = []
    seen = set()
    
    for subcat in subcategories:
        subcat = subcat.strip()
        if not subcat:
            continue
            
        # Map to canonical name
        canonical = subcat_to_canonical.get(subcat, subcat)
        
        if canonical not in seen:
            canonical_subcats.append(canonical)
            seen.add(canonical)
    
    return canonical_subcats

def add_ids_to_taxonomy(taxonomy):
    """Add hierarchical IDs to the taxonomy."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = {"recognition_taxonomy": {}}
    
    for i, (category_name, category_data) in enumerate(taxonomy["recognition_taxonomy"].items()):
        if i >= len(letters):
            print(f"⚠️  Warning: More than {len(letters)} categories, some will not have IDs")
            break
            
        category_id = letters[i]
        
        new_category_data = {
            "id": category_id,
            "description": category_data.get("description", "")
        }
        
        # Add IDs to subcategories
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

def main():
    print("="*70)
    print("Intelligent Taxonomy Compression")
    print("="*70)
    print()
    
    # Load recovered taxonomy
    input_file = "taxonomy_results/recovered_taxonomy.json"
    
    try:
        with open(input_file, 'r') as f:
            taxonomy = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found!")
        print("Please run the recovery script first.")
        return
    
    print(f"📂 Loaded: {input_file}")
    original_count = len(taxonomy['recognition_taxonomy'])
    print(f"   Original categories: {original_count}")
    print()
    
    # Merge similar categories
    print("🔄 Merging similar categories...")
    compressed = merge_similar_categories(taxonomy)
    
    compressed_count = len(compressed['recognition_taxonomy'])
    print(f"✅ Compressed to {compressed_count} categories")
    print(f"   Reduction: {original_count - compressed_count} categories merged")
    print()
    
    # Save compressed taxonomy
    compressed_file = "taxonomy_results/compressed_taxonomy.json"
    with open(compressed_file, 'w') as f:
        json.dump(compressed, f, indent=2)
    print(f"💾 Saved: {compressed_file}")
    print()
    
    # Add IDs
    print("🏷️  Adding hierarchical IDs...")
    taxonomy_with_ids = add_ids_to_taxonomy(compressed)
    
    # Save final taxonomy
    final_file = "taxonomy_results/compressed_taxonomy_with_ids.json"
    with open(final_file, 'w') as f:
        json.dump(taxonomy_with_ids, f, indent=2)
    print(f"💾 Saved: {final_file}")
    print()
    
    # Print summary
    print("="*70)
    print("📊 Final Taxonomy Summary")
    print("="*70)
    
    for cat_name, cat_data in taxonomy_with_ids['recognition_taxonomy'].items():
        num_subs = len(cat_data['subcategories'])
        print(f"\n{cat_data['id']}. {cat_name} ({num_subs} subcategories)")
        print(f"   {cat_data['description'][:80]}...")
        
        # Show first few subcategories
        for i, subcat in enumerate(cat_data['subcategories'][:5]):
            print(f"   {subcat['id']}. {subcat['name']}")
        
        if num_subs > 5:
            print(f"   ... and {num_subs - 5} more")
    
    print()
    print("="*70)
    print("✅ Compression complete!")
    print(f"📁 Final file: {final_file}")
    print("="*70)

if __name__ == '__main__':
    main()