#!/usr/bin/env python3
"""
BF3-Bench Chart Generator v3
Generates three-way comparison charts: DPU vs GPU vs Host CPU

Results (8KB Payload):
  BPE:
    - DPU ARM: 531 µs (C sequential)
    - Host CPU: 2,680 µs (HuggingFace Rust)
    - GPU: 9,235 µs (CUDA sequential)
  WordPiece:
    - DPU ARM: 1,275 µs (HuggingFace Rust)
    - GPU: 1,316 µs (RAPIDS nvtext)
    - Host CPU: 4,756 µs (HuggingFace Rust)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Create output directory
CHART_DIR = '/home/lsalab/bf3-bench/charts'
os.makedirs(CHART_DIR, exist_ok=True)

# Test Results Data (8KB Payload)
RESULTS = {
    'bpe': {
        'DPU ARM': {'time': 531, 'impl': 'C sequential'},
        'Host CPU': {'time': 2680, 'impl': 'HuggingFace (Rust)'},
        'GPU': {'time': 9235, 'impl': 'CUDA sequential'},
    },
    'wordpiece': {
        'DPU ARM': {'time': 1275, 'impl': 'HuggingFace (Rust)'},
        'GPU': {'time': 1316, 'impl': 'RAPIDS nvtext'},
        'Host CPU': {'time': 4756, 'impl': 'HuggingFace (Rust)'},
    }
}

# Pipeline timing (additional stages)
PIPELINE = {
    'rdma_transfer': 1011,  # µs
    'gpu_embedding': 46,    # µs
}

# Color scheme
COLORS = {
    'DPU ARM': '#2196F3',   # Blue
    'Host CPU': '#4CAF50',  # Green
    'GPU': '#FF5722',       # Orange/Red
}

plt.style.use('seaborn-v0_8-whitegrid')


def chart_tokenization_comparison():
    """Bar chart comparing tokenization times across platforms"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (algo, data) in enumerate(RESULTS.items()):
        ax = axes[idx]

        # Sort by time for better visualization
        sorted_items = sorted(data.items(), key=lambda x: x[1]['time'])
        platforms = [p for p, _ in sorted_items]
        times = [d['time'] for _, d in sorted_items]
        colors = [COLORS[p] for p in platforms]

        bars = ax.bar(platforms, times, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                    f'{t:,} µs', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Calculate speedups relative to slowest
        max_time = max(times)
        for bar, t in zip(bars, times):
            if t < max_time:
                speedup = max_time / t
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f'{speedup:.1f}×', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')

        ax.set_ylabel('Tokenization Time (µs)', fontsize=12)
        ax.set_title(f'{algo.upper()} Tokenization\n(8KB Payload)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(times) * 1.2)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Three-Way Tokenization Comparison: DPU vs CPU vs GPU',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/tokenization_comparison_3way.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: tokenization_comparison_3way.png")


def chart_bpe_speedup():
    """BPE speedup comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = RESULTS['bpe']
    platforms = ['DPU ARM', 'Host CPU', 'GPU']
    times = [data[p]['time'] for p in platforms]
    colors = [COLORS[p] for p in platforms]

    # Speedup relative to GPU (slowest)
    gpu_time = data['GPU']['time']
    speedups = [gpu_time / t for t in times]

    bars = ax.bar(platforms, speedups, color=colors, edgecolor='black', linewidth=1.5)

    # Add labels
    for bar, s, t in zip(bars, speedups, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{s:.1f}×\n({t:,} µs)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Speedup vs GPU', fontsize=12)
    ax.set_title('BPE Tokenization Speedup (8KB Payload)\nRelative to GPU Sequential',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(speedups) * 1.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Add annotation
    ax.annotate('DPU is 17× faster than GPU\nand 5× faster than Host CPU',
                xy=(0, speedups[0]), xytext=(0.5, speedups[0] * 0.6),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#2196F3'),
                arrowprops=dict(arrowstyle='->', color='#2196F3'))

    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/bpe_speedup_3way.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: bpe_speedup_3way.png")


def chart_pipeline_timeline():
    """Pipeline timeline comparison for BPE"""
    fig, ax = plt.subplots(figsize=(14, 7))

    scenarios = [
        ('DPU-Based\n(Tokenize on DPU)', 'DPU ARM'),
        ('CPU-Based\n(Tokenize on Host CPU)', 'Host CPU'),
        ('GPU-Based\n(Tokenize on GPU)', 'GPU'),
    ]

    y_positions = [3, 2, 1]
    bar_height = 0.5

    colors = {
        'tokenize': '#2196F3',
        'rdma': '#FF9800',
        'gpu': '#E91E63',
    }

    for (label, platform), y_pos in zip(scenarios, y_positions):
        tok_time = RESULTS['bpe'][platform]['time']
        rdma_time = PIPELINE['rdma_transfer']
        gpu_time = PIPELINE['gpu_embedding']

        x = 0

        # Tokenization
        ax.barh(y_pos, tok_time, left=x, height=bar_height,
                color=colors['tokenize'], edgecolor='white', linewidth=1)
        if tok_time > 300:
            ax.text(x + tok_time/2, y_pos, f'{tok_time:,} µs',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        x += tok_time

        # RDMA Transfer
        ax.barh(y_pos, rdma_time, left=x, height=bar_height,
                color=colors['rdma'], edgecolor='white', linewidth=1)
        ax.text(x + rdma_time/2, y_pos, f'{rdma_time:,} µs',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        x += rdma_time

        # GPU Embedding
        ax.barh(y_pos, gpu_time, left=x, height=bar_height,
                color=colors['gpu'], edgecolor='white', linewidth=1)
        x += gpu_time

        # Total time
        total = tok_time + rdma_time + gpu_time
        ax.text(x + 200, y_pos, f'Total: {total:,} µs',
                ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in scenarios], fontsize=12)
    ax.set_xlabel('Time (µs)', fontsize=12)
    ax.set_title('BPE Pipeline Timeline: DPU vs CPU vs GPU\n(8KB Payload)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 12000)

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors['tokenize'], label='Tokenization'),
        mpatches.Patch(color=colors['rdma'], label='RDMA Transfer'),
        mpatches.Patch(color=colors['gpu'], label='GPU Embedding'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/bpe_pipeline_3way.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: bpe_pipeline_3way.png")


def chart_wordpiece_comparison():
    """WordPiece comparison across platforms"""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = RESULTS['wordpiece']
    # Sort by time
    sorted_items = sorted(data.items(), key=lambda x: x[1]['time'])
    platforms = [p for p, _ in sorted_items]
    times = [d['time'] for _, d in sorted_items]
    impls = [d['impl'] for _, d in sorted_items]
    colors = [COLORS[p] for p in platforms]

    bars = ax.bar(platforms, times, color=colors, edgecolor='black', linewidth=1.5)

    # Add labels with implementation
    for bar, t, impl in zip(bars, times, impls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{t:,} µs\n({impl})', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Tokenization Time (µs)', fontsize=12)
    ax.set_title('WordPiece Tokenization Comparison (8KB Payload)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(times) * 1.25)
    ax.grid(axis='y', alpha=0.3)

    # Annotation
    ax.annotate('DPU and GPU are nearly identical\n(both use optimized implementations)',
                xy=(0.5, min(times)), xytext=(1.5, min(times) + 1500),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'),
                arrowprops=dict(arrowstyle='->', color='#4CAF50'))

    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/wordpiece_comparison_3way.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: wordpiece_comparison_3way.png")


def chart_summary_table():
    """Summary comparison table as image"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Table data
    columns = ['Algorithm', 'DPU ARM', 'Host CPU', 'GPU', 'Best Platform']
    rows = [
        ['BPE (GPT-2)', '531 µs', '2,680 µs', '9,235 µs', 'DPU (17× faster)'],
        ['WordPiece (BERT)', '1,275 µs', '4,756 µs', '1,316 µs', 'DPU/GPU (~equal)'],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#E3F2FD'] * 5,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Highlight best cells
    table[(1, 1)].set_facecolor('#C8E6C9')  # DPU BPE
    table[(2, 1)].set_facecolor('#C8E6C9')  # DPU WordPiece
    table[(2, 3)].set_facecolor('#C8E6C9')  # GPU WordPiece

    plt.title('Tokenization Performance Summary (8KB Payload)',
              fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/summary_table_3way.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: summary_table_3way.png")


def chart_latency_breakdown():
    """Stacked bar chart showing end-to-end latency breakdown"""
    fig, ax = plt.subplots(figsize=(12, 7))

    scenarios = ['DPU-Based', 'CPU-Based', 'GPU-Based']
    platforms = ['DPU ARM', 'Host CPU', 'GPU']

    # BPE times
    tokenize_times = [RESULTS['bpe'][p]['time'] for p in platforms]
    rdma_times = [PIPELINE['rdma_transfer']] * 3
    gpu_times = [PIPELINE['gpu_embedding']] * 3

    x = np.arange(len(scenarios))
    width = 0.5

    colors = {
        'tokenize': '#2196F3',
        'rdma': '#FF9800',
        'gpu': '#E91E63',
    }

    p1 = ax.bar(x, tokenize_times, width, label='Tokenization', color=colors['tokenize'])
    p2 = ax.bar(x, rdma_times, width, bottom=tokenize_times, label='RDMA Transfer', color=colors['rdma'])
    p3 = ax.bar(x, gpu_times, width, bottom=np.array(tokenize_times) + np.array(rdma_times),
                label='GPU Embedding', color=colors['gpu'])

    # Add total labels
    totals = [t + r + g for t, r, g in zip(tokenize_times, rdma_times, gpu_times)]
    for i, total in enumerate(totals):
        ax.text(x[i], total + 200, f'{total:,} µs', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Time (µs)', fontsize=12)
    ax.set_title('End-to-End BPE Pipeline Latency Breakdown (8KB Payload)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/latency_breakdown_3way.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: latency_breakdown_3way.png")


def main():
    print("=" * 60)
    print("BF3-Bench Chart Generator v3 (Three-Way Comparison)")
    print("=" * 60)
    print()

    chart_tokenization_comparison()
    chart_bpe_speedup()
    chart_pipeline_timeline()
    chart_wordpiece_comparison()
    chart_summary_table()
    chart_latency_breakdown()

    print()
    print("=" * 60)
    print(f"All charts generated in: {CHART_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
