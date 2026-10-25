"""Unsmoothed, metric-backed figures plus explicitly theoretical diagrams."""
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .utils import EXPERIMENTS, ROOT, read_history, read_json


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
                         'legend.fontsize': 6.5, 'axes.titlesize': 8, 'lines.linewidth': 1.1,
                         'savefig.dpi': 350, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    for extension in ('png', 'pdf'):
        fig.savefig(target.with_suffix('.' + extension), bbox_inches='tight')
    plt.close(fig)


def theory_figures(root=ROOT):
    style()
    # Endpoints excluded: the Beta(0.2,0.2) density diverges there.
    x = np.linspace(0.001, 0.999, 3000)
    a = 0.2
    log_beta = 2 * math.lgamma(a) - math.lgamma(2 * a)
    density = np.exp((a - 1) * (np.log(x) + np.log1p(-x)) - log_beta)
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    ax.plot(x, density, color='#285775')
    ax.set(xlabel=r'$\lambda$', ylabel='Probability density', xlim=(0, 1), ylim=(0, None))
    ax.text(.5, .85, r'$\mathrm{Beta}(0.2,0.2)$', transform=ax.transAxes, ha='center')
    save(fig, root / 'figures/theory/beta')
    t = np.linspace(0, 300, 1001)
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    ax.plot(t, np.ones_like(t), label='Constant', color='#a86335')
    ax.plot(t, (1 + np.cos(np.pi * t / 300)) / 2, label='Cosine', color='#285775')
    ax.set(xlabel=r'Completed epochs $t$', ylabel=r'$\eta_t / \eta_0$', xlim=(0, 300), ylim=(0, 1.05))
    ax.legend(frameon=False)
    save(fig, root / 'figures/theory/schedule')


def experiment_figures(exp, rows, root=ROOT):
    epochs = [r['epoch'] for r in rows]
    for metric, label in (('loss', 'Cross-entropy'), ('accuracy', 'Accuracy')):
        fig, ax = plt.subplots(figsize=(3.35, 2.15))
        for split, color in (('train', '#285775'), ('val', '#a86335')):
            ax.plot(epochs, [r[f'{split}_{metric}'] for r in rows], label='Training' if split == 'train' else 'Validation', color=color)
        ax.set(xlabel='Epoch', ylabel=label, title=exp)
        ax.set_ylim(0, 1 if metric == 'accuracy' else None)
        ax.legend(frameon=False)
        save(fig, root / f'figures/{exp}/{metric}')
    fig, ax = plt.subplots(figsize=(3.35, 1.8))
    ax.plot(epochs, [r['lr'] for r in rows])
    ax.set(xlabel='Epoch', ylabel='Learning rate used', ylim=(0, None))
    save(fig, root / f'figures/{exp}/learning_rate')


def comparison(name, ids, histories, root=ROOT):
    available = [e for e in ids if e in histories]
    if not available:
        print(f'{name}: no saved metrics; skipped')
        return
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.0))
    for ax, metric in zip(axes.flat, ('train_loss', 'val_loss', 'train_accuracy', 'val_accuracy')):
        for exp in available:
            rows = histories[exp]
            ax.plot([r['epoch'] for r in rows], [r[metric] for r in rows], label=exp)
        ax.set(xlabel='Epoch', ylabel=metric.replace('_', ' ').capitalize())
        ax.set_ylim(0, 1 if 'accuracy' in metric else None)
        ax.legend(frameon=False)
    save(fig, root / f'figures/comparisons/{name}')
    missing = set(ids) - set(available)
    if missing:
        print(f'{name}: partial comparison; missing {sorted(missing)}')


def generate(root=ROOT, theory_only=False):
    theory_figures(root)
    if theory_only:
        return
    histories = {}
    for exp in EXPERIMENTS:
        filename = root / f'results/{exp}/metrics.csv'
        if filename.is_file():
            rows = read_history(filename)
            if rows:
                histories[exp] = rows
                experiment_figures(exp, rows, root)
                print(f'{exp}: plotted {len(rows)} recorded epochs (may be incomplete)')
    comparison('learning_rates', EXPERIMENTS[:3], histories, root)
    comparison('schedules', ('constant', 'cosine'), histories, root)
    comparison('weight_decay', ('cosine', 'wd_5e4', 'wd_1e4'), histories, root)
    baseline_file = root / 'results/selected_baseline.json'
    if baseline_file.is_file():
        baseline = read_json(baseline_file)['experiment']
        comparison('mixup', (baseline, 'mixup'), histories, root)
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    found = False
    for exp in ('cosine', 'wd_5e4', 'wd_1e4'):
        if exp in histories:
            found = True
            rows = histories[exp]
            ax.plot([r['epoch'] for r in rows], [r['train_accuracy'] - r['val_accuracy'] for r in rows], label=exp)
    if found:
        ax.axhline(0, color='gray', linewidth=.5)
        ax.set(xlabel='Epoch', ylabel='Online train - validation accuracy')
        ax.legend(frameon=False)
        save(fig, root / 'figures/comparisons/accuracy_gap')
    else:
        plt.close(fig)
