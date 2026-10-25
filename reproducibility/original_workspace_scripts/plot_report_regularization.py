"""Publication plots from saved epoch metrics, without model execution."""
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'outputs/AI6103_Assignment/results'
OUT = ROOT / 'outputs/AI6103_Report_LaTeX/images'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'lines.linewidth': 1, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

def comparison(name, experiments, logarithmic=False):
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.65))
    for experiment, label, color in experiments:
        with (RESULTS / experiment / 'metrics.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 300
        assert all(math.isfinite(float(row[m])) for row in rows for m in
                   ('train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'))
        for ax, metric in zip(axes.flat, ('train_loss', 'val_loss', 'train_accuracy', 'val_accuracy')):
            ax.plot([int(row['epoch']) for row in rows],
                    [float(row[metric]) * (100 if 'accuracy' in metric else 1) for row in rows],
                    label=label, color=color)
    labels = ('Training cross-entropy' + (' (log scale)' if logarithmic else ''),
              'Validation cross-entropy', 'Training accuracy (%)', 'Validation accuracy (%)')
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel='Epoch', ylabel=label, xlim=(1, 300), xticks=[1, 100, 200, 300])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if logarithmic:
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_ylim(0.001, 10)
    else:
        axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].set_ylim(bottom=0)
    axes[1, 0].set(ylim=(0, 102), yticks=[0, 25, 50, 75, 100])
    axes[1, 1].set(ylim=(0, 42), yticks=[0, 10, 20, 30, 40])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               ncol=len(experiments), frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(pad=.7, rect=(0, 0, 1, .94))
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

comparison('weight_decay', [('cosine', r'$\lambda=0$', '#0072B2'),
                            ('wd_1e4', r'$\lambda=10^{-4}$', '#009E73'),
                            ('wd_5e4', r'$\lambda=5\times10^{-4}$', '#D55E00')], True)
comparison('mixup', [('wd_5e4', 'Baseline', '#0072B2'),
                    ('mixup', r'Mixup ($\alpha=0.2$)', '#D55E00')])

x = np.linspace(.001, .999, 4000)
alpha = .2
log_beta = 2 * math.lgamma(alpha) - math.lgamma(2 * alpha)
y = np.exp((alpha - 1) * np.log(x) + (alpha - 1) * np.log1p(-x) - log_beta)
fig, ax = plt.subplots(figsize=(3.3, 1.7))
ax.plot(x, y, color='#0072B2')
ax.set(xlabel=r'Mixing coefficient $\rho$', ylabel='Probability density',
       xlim=(0, 1), ylim=(0, 30), xticks=[0, .25, .5, .75, 1], yticks=[0, 10, 20, 30])
fig.tight_layout(pad=.6)
fig.savefig(OUT / 'beta.pdf', bbox_inches='tight')
fig.savefig(OUT / 'beta.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('Created weight_decay.pdf, mixup.pdf, beta.pdf from saved results.')
