# coding: utf-8

"""
Compare recorded lip tracking data with viseme mappings
"""

import json
import numpy as np
from typing import Dict
from src.utils.phoneme_to_viseme import EnhancedLivePortraitVisemes


def load_tracking_data(filename: str) -> Dict:
    """Load lip tracking data from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)


def compare_with_visemes(tracking_data: Dict):
    """Compare tracked data with current viseme mappings"""
    print("=" * 80)
    print("COMPARISON: Real Tracked Data vs. Current Viseme Mappings")
    print("=" * 80)

    for phoneme, tracked_params in tracking_data.items():
        # Get current viseme mapping
        viseme = EnhancedLivePortraitVisemes.get_viseme_params(phoneme)
        viseme_dict = viseme.to_dict()

        print(f"\nPhoneme: {phoneme}")
        print("-" * 80)
        print(f"{'Parameter':<30} {'Tracked':<15} {'Current':<15} {'Diff':<15}")
        print("-" * 80)

        for param_name in ['jaw_open', 'lip_width', 'lip_protrusion',
                           'upper_lip_raise', 'lower_lip_lower',
                           'corner_pull_horizontal']:
            tracked_val = tracked_params.get(param_name, 0.0)
            current_val = viseme_dict.get(param_name, 0.0)
            diff = tracked_val - current_val

            # Color code the difference
            status = "✅" if abs(diff) < 0.15 else "⚠️" if abs(diff) < 0.3 else "❌"

            print(f"{param_name:<30} {tracked_val:<15.3f} {current_val:<15.3f} {diff:+.3f} {status}")


def generate_updated_visemes(tracking_data: Dict, output_file: str = "updated_visemes.py"):
    """Generate updated viseme code based on tracked data"""
    print(f"\n🔧 Generating updated viseme mappings to {output_file}...")

    with open(output_file, 'w') as f:
        f.write("# Auto-generated viseme mappings from real lip tracking data\n\n")
        f.write("PHONEME_TO_VISEME_PARAMS = {\n")

        for phoneme, params in tracking_data.items():
            f.write(f"    '{phoneme}': VisemeParams(\n")
            f.write(f"        jaw_open={params.get('jaw_open', 0.0):.3f},\n")
            f.write(f"        lip_width={params.get('lip_width', 0.5):.3f},\n")
            f.write(f"        lip_protrusion={params.get('lip_protrusion', 0.0):.3f},\n")
            f.write(f"        upper_lip_raise={params.get('upper_lip_raise', 0.0):.3f},\n")
            f.write(f"        lower_lip_lower={params.get('lower_lip_lower', 0.0):.3f},\n")
            f.write(f"        corner_pull_horizontal={params.get('corner_pull_horizontal', 0.0):.3f},\n")
            f.write(f"    ),\n")

        f.write("}\n")

    print(f"✅ Updated viseme mappings saved to {output_file}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_lip_tracking.py <tracking_data.json>")
        print("\nExample:")
        print("  python analyze_lip_tracking.py lip_tracking_data_20250115_143000.json")
        sys.exit(1)

    tracking_file = sys.argv[1]

    # Load and compare
    tracking_data = load_tracking_data(tracking_file)
    compare_with_visemes(tracking_data)

    # Generate updated mappings
    generate_updated_visemes(tracking_data)
