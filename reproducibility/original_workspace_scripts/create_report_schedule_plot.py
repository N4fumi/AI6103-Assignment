"""Plot the frozen constant/cosine histories for the report."""
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
                     'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'lines.linewidth': 1.0, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.65))
for experiment, label, color in [('constant', 'Constant', '#0072B2'), ('cosine', 'Cosine', '#D55E00')]:
    with (PROJECT / 'results' / experiment / 'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 300
    for ax, metric in zip(axes.flat, ('train_loss', 'val_loss', 'train_accuracy', 'val_accuracy')):
        ax.plot([int(row['epoch']) for row in rows],
                [float(row[metric]) * (100 if 'accuracy' in metric else 1) for row in rows],
                label=label, color=color)
for ax, label in zip(axes.flat, ('Training cross-entropy (log scale)', 'Validation cross-entropy',
                               'Training accuracy (%)', 'Validation accuracy (%)')):
    ax.set(xlabel='Epoch', ylabel=label, xlim=(1, 300), xticks=[1, 100, 200, 300])
axes[0, 0].set_yscale('log')
axes[0, 0].set_ylim(0.001, 10)
axes[0, 1].set_ylim(bottom=0)
axes[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=4))
axes[1, 0].set_ylim(0, 102)
axes[1, 0].set_yticks([0, 25, 50, 75, 100])
axes[1, 1].set_ylim(0, 22)
axes[1, 1].set_yticks([0, 5, 10, 15, 20])
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2,
           frameon=False, bbox_to_anchor=(.5, 1.02))
fig.tight_layout(pad=.7, rect=(0, 0, 1, .94))
for extension in ('png', 'pdf'):
    fig.savefig(TARGET / f'schedules.{extension}', dpi=350, bbox_inches='tight')
plt.close(fig)
print(TARGET / 'schedules.pdf')
