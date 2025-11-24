# coding: utf-8
"""
Visualize animation metadata from JSON
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_animation(json_path: str):
    """Create visualization from animation metadata JSON"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']

    # Extract data
    timestamps = [f['timestamp'] for f in frames]
    lip_open_values = [f['lip_open'] for f in frames]
    viseme_types = [f['viseme']['type'] if f['viseme'] else 'none' for f in frames]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Lip Open Values
    ax1.plot(timestamps, lip_open_values, linewidth=2, color='#2196F3')
    ax1.fill_between(timestamps, lip_open_values, alpha=0.3, color='#2196F3')
    ax1.set_ylabel('Lip Open (0.0 - 0.8)', fontsize=12)
    ax1.set_title(f"Animation: {data['input_text'][:50]}...", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 0.85)

    # Plot 2: Viseme Timeline
    unique_visemes = sorted(set(viseme_types))
    viseme_to_y = {v: i for i, v in enumerate(unique_visemes)}
    y_values = [viseme_to_y[v] for v in viseme_types]

    ax2.scatter(timestamps, y_values, c=y_values, cmap='tab20', s=20, alpha=0.7)
    ax2.set_ylabel('Viseme Type', fontsize=12)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_yticks(range(len(unique_visemes)))
    ax2.set_yticklabels(unique_visemes)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save plot
    output_path = json_path.replace('_metadata.json', '_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_animation(sys.argv[1])
    else:
        print("Usage: python visualize_animation.py <path_to_metadata.json>")
