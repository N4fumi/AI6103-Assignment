"""Report figure from the frozen 15-epoch measurements; no model execution."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE / 'outputs/AI6103_Assignment'
TARGET = WORKSPACE / 'outputs/report_materials'
TARGET.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
                     'axes.titlesize': 8, 'legend.fontsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'lines.linewidth': 1.2, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
runs = [('lr_05', '0.5', '#0072B2'), ('lr_005', '0.05', '#D55E00'), ('lr_0001', '0.001', '#009E73')]
metrics = [('train_loss', 'Training cross-entropy'), ('val_loss', 'Validation cross-entropy'),
           ('train_accuracy', 'Training accuracy (%)'), ('val_accuracy', 'Validation accuracy (%)')]
fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.65))
for experiment, rate, color in runs:
    with (PROJECT / 'results' / experiment / 'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15
    for ax, (metric, label) in zip(axes.flat, metrics):
        factor = 100 if 'accuracy' in metric else 1
        ax.plot([int(row['epoch']) for row in rows], [factor*float(row[metric]) for row in rows],
                label=rf'$\eta={rate}$', color=color)
for ax, (_, label) in zip(axes.flat, metrics):
    ax.set(xlabel='Epoch', ylabel=label, xlim=(1, 15), xticks=[1, 5, 10, 15])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
axes[0, 0].set_ylim(4.0, 5.15)
axes[0, 1].set_ylim(4.1, 5.0)
axes[1, 0].set_ylim(0, 7)
axes[1, 1].set_ylim(0, 5.5)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3,
           frameon=False, bbox_to_anchor=(.5, 1.02))
fig.tight_layout(pad=.7, rect=(0, 0, 1, .94))
for extension in ('png', 'pdf'):
    fig.savefig(TARGET / f'learning_rates.{extension}', dpi=350, bbox_inches='tight')
plt.close(fig)
print(TARGET / 'learning_rates.pdf')
