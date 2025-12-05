"""
Radial sensitivity plot for Sobol sensitivity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES_DIR, PARAMETER_GROUPS

DPI_HIGH = 300

def get_parameter_display_names():
    return {
        'mrf_cannonsville': 'MRF' + chr(10) + 'Cannonsville',
        'mrf_pepacton': 'MRF' + chr(10) + 'Pepacton',
        'mrf_neversink': 'MRF' + chr(10) + 'Neversink',
        'mrf_summer_start_shift': 'Summer' + chr(10) + 'Start Shift',
        'mrf_fall_start_shift': 'Fall' + chr(10) + 'Start Shift',
        'mrf_winter_start_shift': 'Winter' + chr(10) + 'Start Shift',
        'mrf_spring_start_shift': 'Spring' + chr(10) + 'Start Shift',
        'mrf_summer_scale': 'Summer' + chr(10) + 'Scale',
        'mrf_fall_scale': 'Fall' + chr(10) + 'Scale',
        'mrf_winter_scale': 'Winter' + chr(10) + 'Scale',
        'mrf_spring_scale': 'Spring' + chr(10) + 'Scale',
        'zone_level1b_vertical_shift': 'Zone 1b' + chr(10) + 'Vertical',
        'zone_level1c_vertical_shift': 'Zone 1c' + chr(10) + 'Vertical',
        'zone_level2_vertical_shift': 'Zone 2' + chr(10) + 'Vertical',
        'zone_level3_vertical_shift': 'Zone 3' + chr(10) + 'Vertical',
        'zone_level4_vertical_shift': 'Zone 4' + chr(10) + 'Vertical',
        'zone_level5_vertical_shift': 'Zone 5' + chr(10) + 'Vertical',
        'zone_level1b_time_shift': 'Zone 1b' + chr(10) + 'Time',
        'zone_level1c_time_shift': 'Zone 1c' + chr(10) + 'Time',
        'zone_level2_time_shift': 'Zone 2' + chr(10) + 'Time',
        'zone_level3_time_shift': 'Zone 3' + chr(10) + 'Time',
        'zone_level4_time_shift': 'Zone 4' + chr(10) + 'Time',
        'zone_level5_time_shift': 'Zone 5' + chr(10) + 'Time',
    }


def plot_radial_sensitivity(sobol_results, metric, figsize=(12, 14), title=None,
                           marker_color='#8B0000', ring_color='#8B0000',
                           line_color='#1f3b73', s1_scale=3000, st_scale=0.15,
                           s2_scale=15, s2_threshold=0.01, save=True,
                           filename=None, show_legend=True):
    if metric not in sobol_results:
        raise ValueError(f"Metric '{metric}' not found")
    indices = sobol_results[metric]
    if 'error' in indices:
        raise ValueError(f"Error for {metric}: {indices['error']}")

    param_names = indices['parameter_names']
    S1 = np.nan_to_num(np.array(indices['S1']), nan=0.0)
    ST = np.nan_to_num(np.array(indices['ST']), nan=0.0)
    S2 = np.nan_to_num(np.array(indices.get('S2', np.zeros((len(param_names), len(param_names))))), nan=0.0)
    S1 = np.maximum(S1, 0)
    ST = np.maximum(ST, 0)

    display_names = get_parameter_display_names()
    n_params = len(param_names)
    angles = np.linspace(0, 2*np.pi, n_params, endpoint=False) - np.pi/2
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'aspect': 'equal'})

    for i in range(n_params):
        for j in range(i+1, n_params):
            if abs(S2[i,j]) > s2_threshold:
                ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                       color=line_color, linewidth=abs(S2[i,j])*s2_scale, alpha=0.7, zorder=1)

    for i, param in enumerate(param_names):
        if ST[i] > 0:
            ring = Circle((x_pos[i], y_pos[i]), st_scale*np.sqrt(ST[i]),
                         fill=False, edgecolor=ring_color, linewidth=2.5, zorder=2)
            ax.add_patch(ring)
        ax.scatter(x_pos[i], y_pos[i], s=max(S1[i]*s1_scale, 20),
                  c=marker_color, edgecolors='white', linewidths=1, zorder=3)
        lx, ly = 1.25*np.cos(angles[i]), 1.25*np.sin(angles[i])
        deg = np.degrees(angles[i]) % 360
        ha = 'center' if 45<=deg<135 or 225<=deg<315 else ('right' if 135<=deg<225 else 'left')
        va = 'bottom' if 45<=deg<135 else ('top' if 225<=deg<315 else 'center')
        ax.text(lx, ly, display_names.get(param, param.replace('_', chr(10))), ha=ha, va=va, fontsize=9)

    ax.set_title(title or metric.replace('_', ' ').title(), fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis('off')

    if show_legend:
        _add_legend(fig, S1, ST, S2, s1_scale, st_scale, s2_scale, marker_color, ring_color, line_color)

    plt.tight_layout()
    if save:
        filepath = FIGURES_DIR / f"{filename or f'radial_sensitivity_{metric}'}.png"
        plt.savefig(filepath, dpi=DPI_HIGH, bbox_inches='tight', facecolor='white')
        print(f'Saved: {filepath}')
    return fig, ax


def _add_legend(fig, S1, ST, S2, s1_scale, st_scale, s2_scale, marker_color, ring_color, line_color):
    s1_max = max(np.nanmax(S1), 0.01)
    st_max = max(np.nanmax(ST), 0.01)
    s2_flat = S2[np.triu_indices_from(S2, k=1)]
    s2_max = max(np.nanmax(np.abs(s2_flat)), 0.01)

    leg = fig.add_axes([0.1, 0.02, 0.8, 0.18])
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.axis('off')

    leg.text(0.12, 0.95, 'First-Order', ha='center', va='top', fontsize=10, fontweight='bold')
    for i, v in enumerate([s1_max, s1_max/2]):
        leg.scatter(0.05+i*0.12, 0.45, s=v*s1_scale*0.15, c=marker_color, edgecolors='white')
        leg.text(0.05+i*0.12, 0.15, f'{v*100:.0f}%', ha='center', fontsize=8)

    leg.text(0.5, 0.95, 'Total-Order', ha='center', va='top', fontsize=10, fontweight='bold')
    for i, v in enumerate([st_max, st_max/2]):
        leg.add_patch(Circle((0.42+i*0.14, 0.45), st_scale*np.sqrt(v)*0.4, fill=False, edgecolor=ring_color, linewidth=2))
        leg.text(0.42+i*0.14, 0.15, f'{v*100:.0f}%', ha='center', fontsize=8)

    leg.text(0.85, 0.95, 'Second-Order', ha='center', va='top', fontsize=10, fontweight='bold')
    for i, v in enumerate([s2_max, s2_max/2]):
        leg.plot([0.75, 0.9], [0.55-i*0.2]*2, color=line_color, linewidth=v*s2_scale, alpha=0.7)
        leg.text(0.93, 0.55-i*0.2, f'{v*100:.0f}%', ha='left', va='center', fontsize=8)


def generate_radial_figures(sobol_results, metrics=None):
    import matplotlib
    matplotlib.use('Agg')
    metrics = metrics or [m for m in sobol_results if 'error' not in sobol_results.get(m, {'error': True})]
    for m in metrics:
        try:
            fig, _ = plot_radial_sensitivity(sobol_results, m, save=True)
            plt.close(fig)
        except Exception as e:
            print(f'Error plotting {m}: {e}')


if __name__ == '__main__':
    from methods.analysis import load_sobol_results
    try:
        _, raw = load_sobol_results()
        generate_radial_figures(raw)
    except FileNotFoundError:
        print('No Sobol results found.')
