"""
This script searches through all Python files in the layouts directory
to identify and fix React/Dash inline style issues (hyphenated CSS properties).

Run this script from your project root directory:
python search_fix_styles.py
"""
import os
import re
import sys

def convert_to_camel_case(match):
    """Convert hyphenated CSS property to camelCase."""
    key = match.group(1)
    if '-' in key:
        # Convert hyphenated to camelCase
        parts = key.split('-')
        key = parts[0] + ''.join(part.capitalize() for part in parts[1:])
    return f'"{key}"'

def process_file(filepath):
    """Process a file to identify and fix style issues."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Search for any style dictionaries with hyphenated properties
    style_pattern = r'style=\{([^}]+)\}'
    style_dicts = re.findall(style_pattern, content)
    
    has_issues = False
    for style_dict in style_dicts:
        # Look for hyphenated property names
        prop_pattern = r'"([a-zA-Z-]+)"'
        props = re.findall(prop_pattern, style_dict)
        for prop in props:
            if '-' in prop:
                has_issues = True
                print(f"  - Found hyphenated property: {prop}")
    
    if has_issues:
        # Fix the style issues by converting hyphenated properties to camelCase
        fixed_content = re.sub(r'"([a-zA-Z-]+)"', convert_to_camel_case, content)
        
        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"  ✓ Fixed style issues in {filepath}")
        return True
    
    return False

def main():
    """Main function to scan and fix files."""
    # Directory to scan - change to 'layouts' to target only layout files
    scan_dir = 'layouts'
    fixed_count = 0
    
    if not os.path.exists(scan_dir):
        print(f"Error: Directory '{scan_dir}' not found. Run this script from your project root.")
        return
    
    print(f"Scanning {scan_dir} directory for React/Dash style issues...")
    
    # Walk through all .py files in the directory
    for root, dirs, files in os.walk(scan_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                print(f"Checking {filepath}...")
                if process_file(filepath):
                    fixed_count += 1
    
    print(f"\nFixed style issues in {fixed_count} files.")
    if fixed_count > 0:
        print("Please restart your Dash application to apply the changes.")

if __name__ == "__main__":
    main()