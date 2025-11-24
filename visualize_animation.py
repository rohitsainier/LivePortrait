# coding: utf-8
"""
🎨 Enhanced Animation Metadata Visualizer
Visualizes:
- Lip open values with intensity
- Viseme timeline with stress markers
- Phoneme duration analysis
- Coarticulation effects
- Statistical quality metrics
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


class AnimationVisualizer:
    """Enhanced visualizer for animation metadata"""

    # Color scheme for viseme categories
    VISEME_COLORS = {
        # Silence
        'sil': '#9E9E9E',      # Gray
        'none': '#BDBDBD',     # Light gray

        # Consonants - Closures
        'PP': '#F44336',       # Red

        # Consonants - Fricatives
        'FF': '#E91E63',       # Pink
        'TH': '#EC407A',
        'SS': '#D81B60',
        'CH': '#C2185B',

        # Consonants - Stops
        'DD': '#9C27B0',       # Purple
        'kk': '#7B1FA2',

        # Consonants - Nasals & Liquids
        'nn': '#673AB7',       # Deep Purple
        'L': '#5E35B1',
        'RR': '#512DA8',
        'WQ': '#4527A0',

        # Vowels - Close
        'I': '#2196F3',        # Blue
        'U': '#1976D2',

        # Vowels - Mid
        'E': '#03A9F4',        # Light Blue
        'ER': '#0288D1',
        'EI': '#0277BD',

        # Vowels - Open
        'O': '#00BCD4',        # Cyan
        'aa': '#0097A7',

        # Diphthongs
        'AI': '#009688',       # Teal
        'AW': '#00897B',
        'OW': '#00796B',
    }

    def __init__(self, json_path: str):
        """Load animation metadata"""
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.frames = self.data['frames']
        self.config = self.data.get('configuration', {})
        self.stats = self.data.get('statistics', {})

        # Extract frame data
        self.timestamps = [f['timestamp'] for f in self.frames]
        self.lip_open_values = [f['lip_open'] for f in self.frames]
        self.viseme_types = [f.get('viseme', {}).get('type', 'none') for f in self.frames]
        self.phonemes = [f.get('viseme', {}).get('phoneme', '') for f in self.frames]
        self.intensities = [f.get('viseme', {}).get('intensity', 1.0) for f in self.frames]

        # Parse viseme info
        self.visemes = []
        for f in self.frames:
            v_info = f.get('viseme')
            if v_info and v_info.get('type') != 'none':
                self.visemes.append({
                    'type': v_info.get('type', 'sil'),
                    'phoneme': v_info.get('phoneme', ''),
                    'start': v_info.get('start_time', 0),
                    'end': v_info.get('end_time', 0),
                    'intensity': v_info.get('intensity', 1.0),
                })

    def plot_comprehensive(self, output_path: str = None):
        """Create comprehensive 4-panel visualization"""

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)

        # Panel 1: Lip Open Values with Intensity
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_lip_open_with_intensity(ax1)

        # Panel 2: Viseme Timeline
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_viseme_timeline(ax2)

        # Panel 3: Duration Analysis
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_duration_distribution(ax3)

        # Panel 4: Viseme Category Distribution
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_viseme_distribution(ax4)

        # Panel 5: Quality Metrics
        ax5 = fig.add_subplot(gs[3, :])
        self._plot_quality_metrics(ax5)

        # Overall title
        title = self.data.get('input_text', 'Animation')
        if len(title) > 80:
            title = title[:77] + '...'

        fig.suptitle(
            f"Animation Analysis: {title}",
            fontsize=16,
            fontweight='bold',
            y=0.995
        )

        # Save
        if output_path is None:
            json_path = Path(self.data.get('output_video', 'output.mp4'))
            output_path = json_path.with_name(json_path.stem + '_analysis.png')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comprehensive visualization saved to: {output_path}")

        return fig

    def _plot_lip_open_with_intensity(self, ax):
        """Plot lip_open values with intensity coloring"""

        # Create color map based on intensity
        colors = plt.cm.RdYlGn([min(1.0, i / 1.5) for i in self.intensities])

        # Plot line
        ax.plot(self.timestamps, self.lip_open_values,
                linewidth=2, color='#1976D2', alpha=0.7, zorder=2)

        # Fill under curve with intensity colors
        for i in range(len(self.timestamps) - 1):
            ax.fill_between(
                self.timestamps[i:i+2],
                self.lip_open_values[i:i+2],
                alpha=0.4,
                color=colors[i],
                zorder=1
            )

        # Mark stressed visemes
        stressed_frames = [
            (self.timestamps[i], self.lip_open_values[i])
            for i, f in enumerate(self.frames)
            if f.get('viseme', {}).get('intensity', 1.0) > 1.1
        ]

        if stressed_frames:
            stressed_t, stressed_v = zip(*stressed_frames)
            ax.scatter(stressed_t, stressed_v,
                      color='red', s=50, marker='*',
                      label='Stressed', zorder=3, alpha=0.8)

        ax.set_ylabel('Lip Open (0.0 - 0.8)', fontsize=11, fontweight='bold')
        ax.set_title('Lip Opening Timeline (colored by intensity)', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 0.85)
        ax.legend(loc='upper right')

        # Add reference lines
        ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

        ax.text(ax.get_xlim()[1] * 0.98, 0.2, 'consonants',
                ha='right', va='bottom', fontsize=8, alpha=0.6)
        ax.text(ax.get_xlim()[1] * 0.98, 0.5, 'open vowels',
                ha='right', va='bottom', fontsize=8, alpha=0.6)

    def _plot_viseme_timeline(self, ax):
        """Plot viseme timeline with phoneme labels"""

        unique_visemes = sorted(set(v['type'] for v in self.visemes))
        viseme_to_y = {v: i for i, v in enumerate(unique_visemes)}

        # Plot viseme blocks
        for viseme in self.visemes:
            v_type = viseme['type']
            start = viseme['start']
            duration = viseme['end'] - viseme['start']
            y = viseme_to_y[v_type]

            color = self.VISEME_COLORS.get(v_type, '#757575')

            # Draw rectangle
            rect = mpatches.Rectangle(
                (start, y - 0.4), duration, 0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)

            # Add phoneme label if space allows
            if duration > 0.05:  # Only label if wide enough
                ax.text(
                    start + duration / 2, y,
                    viseme['phoneme'],
                    ha='center', va='center',
                    fontsize=7,
                    fontweight='bold',
                    color='white' if v_type not in ['sil', 'none'] else 'black'
                )

        ax.set_ylabel('Viseme Type', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(unique_visemes)))
        ax.set_yticklabels(unique_visemes, fontsize=9)
        ax.set_title('Viseme Timeline (with phoneme labels)', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_xlim(0, max(self.timestamps))

    def _plot_duration_distribution(self, ax):
        """Plot phoneme duration distribution"""

        # Calculate durations
        durations = []
        labels = []

        for viseme in self.visemes:
            if viseme['type'] != 'sil':
                duration_ms = (viseme['end'] - viseme['start']) * 1000
                durations.append(duration_ms)
                labels.append(f"{viseme['phoneme']} ({viseme['type']})")

        if not durations:
            ax.text(0.5, 0.5, 'No duration data',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Sort by duration
        sorted_data = sorted(zip(durations, labels), reverse=True)[:15]  # Top 15
        durations, labels = zip(*sorted_data)

        # Create horizontal bar chart
        colors = [self.VISEME_COLORS.get(l.split('(')[1].rstrip(')'), '#757575')
                 for l in labels]

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, durations, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Duration (ms)', fontsize=10, fontweight='bold')
        ax.set_title('Top 15 Longest Phonemes', fontsize=11, pad=10)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')

        # Add value labels
        for i, (bar, dur) in enumerate(zip(bars, durations)):
            ax.text(dur + 2, i, f'{dur:.1f}',
                   va='center', fontsize=7, fontweight='bold')

    def _plot_viseme_distribution(self, ax):
        """Plot viseme category distribution"""

        # Count viseme types
        viseme_counts = Counter(v['type'] for v in self.visemes if v['type'] != 'sil')

        if not viseme_counts:
            ax.text(0.5, 0.5, 'No viseme data',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Sort by count
        sorted_visemes = sorted(viseme_counts.items(), key=lambda x: x[1], reverse=True)
        visemes, counts = zip(*sorted_visemes)

        # Colors
        colors = [self.VISEME_COLORS.get(v, '#757575') for v in visemes]

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=visemes,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 8, 'fontweight': 'bold'}
        )

        # Style percentage labels
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(7)

        ax.set_title('Viseme Distribution (excluding silence)', fontsize=11, pad=10)

    def _plot_quality_metrics(self, ax):
        """Display quality metrics and statistics"""

        ax.axis('off')

        # Calculate metrics
        metrics = self._calculate_quality_metrics()

        # Create text layout
        metric_text = "📊 ANIMATION QUALITY METRICS\n\n"

        # Basic info
        metric_text += f"{'Configuration':<25}\n"
        metric_text += f"  • Total Frames: {self.stats.get('total_frames', len(self.frames))}\n"
        metric_text += f"  • Duration: {self.stats.get('duration_seconds', 0):.2f}s\n"
        metric_text += f"  • FPS: {self.config.get('fps', 25)}\n"
        metric_text += f"  • Smoothing: {self.config.get('smoothing', 0.15):.2f}\n"
        metric_text += f"  • Stitching: {self.config.get('flag_stitching', False)}\n\n"

        # Viseme stats
        metric_text += f"{'Viseme Statistics':<25}\n"
        metric_text += f"  • Total Visemes: {len(self.visemes)}\n"
        metric_text += f"  • Unique Types: {metrics['unique_visemes']}\n"
        metric_text += f"  • Avg Duration: {metrics['avg_duration_ms']:.1f}ms\n"
        metric_text += f"  • Min Duration: {metrics['min_duration_ms']:.1f}ms\n"
        metric_text += f"  • Max Duration: {metrics['max_duration_ms']:.1f}ms\n\n"

        # Lip movement stats
        metric_text += f"{'Lip Movement Analysis':<25}\n"
        metric_text += f"  • Avg Lip Open: {metrics['avg_lip_open']:.3f}\n"
        metric_text += f"  • Max Lip Open: {metrics['max_lip_open']:.3f}\n"
        metric_text += f"  • Movement Range: {metrics['lip_range']:.3f}\n"
        metric_text += f"  • Stressed Frames: {metrics['stressed_count']}\n\n"

        # Quality score
        quality_score = self._calculate_quality_score(metrics)
        quality_color = 'green' if quality_score >= 8 else 'orange' if quality_score >= 6 else 'red'

        metric_text += f"{'Quality Assessment':<25}\n"
        metric_text += f"  • Overall Score: {quality_score:.1f}/10.0 "

        ax.text(0.05, 0.95, metric_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Quality indicator
        quality_label = "EXCELLENT" if quality_score >= 8 else "GOOD" if quality_score >= 6 else "NEEDS IMPROVEMENT"
        ax.text(0.95, 0.5, quality_label,
               transform=ax.transAxes,
               fontsize=20,
               fontweight='bold',
               color=quality_color,
               ha='right',
               va='center',
               bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.2, pad=1))

    def _calculate_quality_metrics(self) -> Dict:
        """Calculate quality metrics"""

        durations = [(v['end'] - v['start']) * 1000 for v in self.visemes if v['type'] != 'sil']

        return {
            'unique_visemes': len(set(v['type'] for v in self.visemes)),
            'avg_duration_ms': np.mean(durations) if durations else 0,
            'min_duration_ms': min(durations) if durations else 0,
            'max_duration_ms': max(durations) if durations else 0,
            'avg_lip_open': np.mean(self.lip_open_values),
            'max_lip_open': max(self.lip_open_values),
            'lip_range': max(self.lip_open_values) - min(self.lip_open_values),
            'stressed_count': sum(1 for i in self.intensities if i > 1.1),
        }

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score (0-10)"""

        score = 5.0  # Base score

        # Good duration variety (+2 points)
        if 40 < metrics['avg_duration_ms'] < 120:
            score += 2.0
        elif 30 < metrics['avg_duration_ms'] < 150:
            score += 1.0

        # Good lip movement range (+2 points)
        if metrics['lip_range'] > 0.5:
            score += 2.0
        elif metrics['lip_range'] > 0.3:
            score += 1.0

        # Good viseme variety (+1 point)
        if metrics['unique_visemes'] >= 10:
            score += 1.0

        # Stress markers present (+1 point)
        if metrics['stressed_count'] > 0:
            score += 0.5

        return min(10.0, score)

    def export_statistics(self, output_path: str = None):
        """Export detailed statistics to JSON"""

        if output_path is None:
            json_path = Path(self.data.get('output_video', 'output.mp4'))
            output_path = json_path.with_name(json_path.stem + '_statistics.json')

        metrics = self._calculate_quality_metrics()
        quality_score = self._calculate_quality_score(metrics)

        stats = {
            'quality_score': round(quality_score, 2),
            'metrics': metrics,
            'viseme_counts': dict(Counter(v['type'] for v in self.visemes)),
            'phoneme_counts': dict(Counter(v['phoneme'] for v in self.visemes)),
            'duration_percentiles': {
                '25th': np.percentile([v['end'] - v['start'] for v in self.visemes], 25) * 1000,
                '50th': np.percentile([v['end'] - v['start'] for v in self.visemes], 50) * 1000,
                '75th': np.percentile([v['end'] - v['start'] for v in self.visemes], 75) * 1000,
                '95th': np.percentile([v['end'] - v['start'] for v in self.visemes], 95) * 1000,
            }
        }

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Statistics exported to: {output_path}")

        return stats


def visualize_animation(json_path: str, show_plot: bool = True):
    """
    Main function to visualize animation metadata

    Args:
        json_path: Path to animation metadata JSON
        show_plot: Whether to display the plot
    """

    visualizer = AnimationVisualizer(json_path)

    # Create comprehensive visualization
    fig = visualizer.plot_comprehensive()

    # Export statistics
    visualizer.export_statistics()

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def compare_animations(json_paths: List[str], output_path: str = 'comparison.png'):
    """
    Compare multiple animations side-by-side

    Args:
        json_paths: List of metadata JSON paths
        output_path: Output comparison image path
    """

    n_animations = len(json_paths)
    fig, axes = plt.subplots(n_animations, 1, figsize=(14, 4 * n_animations))

    if n_animations == 1:
        axes = [axes]

    for idx, json_path in enumerate(json_paths):
        visualizer = AnimationVisualizer(json_path)

        ax = axes[idx]

        # Plot lip open values
        ax.plot(visualizer.timestamps, visualizer.lip_open_values,
               linewidth=2, label=Path(json_path).stem)
        ax.fill_between(visualizer.timestamps, visualizer.lip_open_values,
                        alpha=0.3)
        ax.set_ylabel('Lip Open', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.85)

    fig.suptitle('Animation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single animation:   python visualize_animation.py <metadata.json>")
        print("  Compare animations: python visualize_animation.py <json1> <json2> ... --compare")
        sys.exit(1)

    if '--compare' in sys.argv:
        json_files = [arg for arg in sys.argv[1:] if arg != '--compare']
        compare_animations(json_files)
    else:
        visualize_animation(sys.argv[1], show_plot=True)
