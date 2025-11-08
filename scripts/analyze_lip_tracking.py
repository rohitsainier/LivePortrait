# coding: utf-8

"""
Compare recorded lip tracking data with viseme mappings
Works with both webcam recordings and video analysis
"""

import json
import numpy as np
from typing import Dict, List
from pathlib import Path
from collections import defaultdict


def load_tracking_data(filename: str) -> Dict:
    """Load lip tracking data from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)


def compare_with_visemes(tracking_data: Dict):
    """Compare tracked data with current viseme mappings"""
    # Import here to avoid circular dependency
    from src.utils.phoneme_to_viseme import EnhancedLivePortraitVisemes

    print("=" * 80)
    print("COMPARISON: Real Tracked Data vs. Current Viseme Mappings")
    print("=" * 80)

    all_diffs = []

    for phoneme, tracked_params in sorted(tracking_data.items()):
        # Get current viseme mapping
        viseme = EnhancedLivePortraitVisemes.get_viseme_params(phoneme)
        viseme_dict = viseme.to_dict()

        print(f"\nPhoneme: {phoneme}")
        print("-" * 80)
        print(f"{'Parameter':<30} {'Tracked':<15} {'Current':<15} {'Diff':<15}")
        print("-" * 80)

        phoneme_diffs = []

        for param_name in ['jaw_open', 'lip_width', 'lip_protrusion',
                           'upper_lip_raise', 'lower_lip_lower',
                           'corner_pull_horizontal']:
            tracked_val = tracked_params.get(param_name, 0.0)
            current_val = viseme_dict.get(param_name, 0.0)
            diff = tracked_val - current_val

            phoneme_diffs.append(abs(diff))

            # Color code the difference
            if abs(diff) < 0.10:
                status = "✅ Good"
            elif abs(diff) < 0.20:
                status = "⚠️  Check"
            else:
                status = "❌ Large diff"

            print(f"{param_name:<30} {tracked_val:<15.3f} {current_val:<15.3f} {diff:+.3f} {status}")

        # Calculate average difference for this phoneme
        avg_diff = np.mean(phoneme_diffs)
        all_diffs.append(avg_diff)
        print(f"\n{'Average absolute difference:':<30} {avg_diff:.3f}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total phonemes analyzed: {len(tracking_data)}")
    print(f"Average absolute difference (all phonemes): {np.mean(all_diffs):.3f}")
    print(f"Max difference: {np.max(all_diffs):.3f}")
    print(f"Min difference: {np.min(all_diffs):.3f}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if np.mean(all_diffs) < 0.15:
        print("✅ Your current viseme mappings are quite accurate!")
        print("   Minor adjustments recommended for phonemes with ❌ markers")
    elif np.mean(all_diffs) < 0.25:
        print("⚠️  Moderate differences detected")
        print("   Consider updating viseme mappings from the generated file")
    else:
        print("❌ Significant differences detected")
        print("   Strongly recommend updating viseme mappings with tracked values")


def generate_updated_visemes(tracking_data: Dict, output_file: str = "updated_visemes.py"):
    """Generate updated viseme code based on tracked data"""
    print(f"\n🔧 Generating updated viseme mappings to {output_file}...")

    with open(output_file, 'w') as f:
        f.write("# coding: utf-8\n\n")
        f.write('"""\n')
        f.write("Auto-generated viseme mappings from real lip tracking data\n")
        f.write(f"Generated from: {len(tracking_data)} phonemes\n")
        f.write('"""\n\n')
        f.write("from dataclasses import dataclass\n\n")
        f.write("@dataclass\n")
        f.write("class VisemeParams:\n")
        f.write("    jaw_open: float = 0.0\n")
        f.write("    lip_width: float = 0.5\n")
        f.write("    lip_protrusion: float = 0.0\n")
        f.write("    upper_lip_raise: float = 0.0\n")
        f.write("    lower_lip_lower: float = 0.0\n")
        f.write("    corner_pull_horizontal: float = 0.0\n")
        f.write("    corner_pull_back: float = 0.0\n")
        f.write("    lip_tightness: float = 0.5\n\n")

        f.write("# Phoneme to Viseme Parameter Mappings\n")
        f.write("PHONEME_TO_VISEME_PARAMS = {\n")

        for phoneme in sorted(tracking_data.keys()):
            params = tracking_data[phoneme]
            f.write(f"    '{phoneme}': VisemeParams(\n")
            f.write(f"        jaw_open={params.get('jaw_open', 0.0):.3f},\n")
            f.write(f"        lip_width={params.get('lip_width', 0.5):.3f},\n")
            f.write(f"        lip_protrusion={params.get('lip_protrusion', 0.0):.3f},\n")
            f.write(f"        upper_lip_raise={params.get('upper_lip_raise', 0.0):.3f},\n")
            f.write(f"        lower_lip_lower={params.get('lower_lip_lower', 0.0):.3f},\n")
            f.write(f"        corner_pull_horizontal={params.get('corner_pull_horizontal', 0.0):.3f},\n")
            f.write(f"        corner_pull_back={params.get('corner_pull_back', 0.0):.3f},\n")
            f.write(f"        lip_tightness={params.get('lip_tightness', 0.5):.3f},\n")
            f.write(f"    ),\n")

        f.write("}\n")

    print(f"✅ Updated viseme mappings saved to {output_file}")
    print(f"\n📋 To apply these values:")
    print(f"   1. Open: src/utils/phoneme_to_viseme.py")
    print(f"   2. Update PHONEME_TO_VISEME_PARAMS with values from {output_file}")
    print(f"   3. Test with: python app.py")


def merge_multiple_analyses(file_list: List[str], output_file: str = "merged_analysis.json"):
    """Merge multiple analysis JSON files by averaging"""
    print(f"\n🔗 Merging {len(file_list)} analysis files...")

    all_data = defaultdict(list)

    # Load all files
    for filename in file_list:
        data = load_tracking_data(filename)
        for phoneme, params in data.items():
            all_data[phoneme].append(params)

    # Average parameters
    merged = {}
    for phoneme, param_list in all_data.items():
        avg_params = {}
        for key in param_list[0].keys():
            values = [p[key] for p in param_list]
            avg_params[key] = float(np.mean(values))
        merged[phoneme] = avg_params
        print(f"  {phoneme}: Averaged {len(param_list)} samples")

    # Save merged data
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"✅ Merged data saved to: {output_file}")
    return merged


if __name__ == '__main__':
    import sys
    from collections import defaultdict

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:   python analyze_lip_tracking.py <tracking_data.json>")
        print("  Multiple files: python analyze_lip_tracking.py <file1.json> <file2.json> ...")
        print("\nExample:")
        print("  python analyze_lip_tracking.py lip_analysis_video1.json")
        print("  python analyze_lip_tracking.py video1.json video2.json video3.json")
        sys.exit(1)

    tracking_files = sys.argv[1:]

    # Check if files exist
    for f in tracking_files:
        if not Path(f).exists():
            print(f"❌ File not found: {f}")
            sys.exit(1)

    # If multiple files, merge them first
    if len(tracking_files) > 1:
        print(f"📊 Multiple files detected: {len(tracking_files)}")
        tracking_data = merge_multiple_analyses(tracking_files)
    else:
        tracking_data = load_tracking_data(tracking_files[0])

    # Compare with current viseme mappings
    compare_with_visemes(tracking_data)

    # Generate updated mappings
    generate_updated_visemes(tracking_data)
