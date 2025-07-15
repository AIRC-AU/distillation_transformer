#!/usr/bin/env python3
"""
Script to combine separate German and Chinese Bible files into a single tab-separated file
for the German-Chinese translation model.
"""

def create_combined_dataset():
    """Combine German and Chinese files into tab-separated format."""
    
    # File paths
    german_file = "bible-uedin.de-zh.de"
    chinese_file = "bible-uedin.de-zh.zh"
    output_file = "de-zh.txt"
    
    print("Reading German and Chinese files...")
    
    # Read German sentences
    with open(german_file, 'r', encoding='utf-8') as f:
        german_lines = [line.strip() for line in f.readlines()]
    
    # Read Chinese sentences  
    with open(chinese_file, 'r', encoding='utf-8') as f:
        chinese_lines = [line.strip() for line in f.readlines()]
    
    # Verify both files have same number of lines
    if len(german_lines) != len(chinese_lines):
        print(f"Warning: German file has {len(german_lines)} lines, Chinese file has {len(chinese_lines)} lines")
        min_lines = min(len(german_lines), len(chinese_lines))
        german_lines = german_lines[:min_lines]
        chinese_lines = chinese_lines[:min_lines]
        print(f"Using first {min_lines} lines from both files")
    
    print(f"Creating combined dataset with {len(german_lines)} sentence pairs...")
    
    # Create combined file
    with open(output_file, 'w', encoding='utf-8') as f:
        for german, chinese in zip(german_lines, chinese_lines):
            # Skip empty lines
            if german.strip() and chinese.strip():
                f.write(f"{german}\t{chinese}\n")
    
    print(f"Combined dataset saved as {output_file}")
    print("Sample entries:")
    
    # Show first 5 entries
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                print(f"{i+1}. DE: {parts[0]}")
                print(f"   ZH: {parts[1]}")
                print()

if __name__ == "__main__":
    create_combined_dataset()
