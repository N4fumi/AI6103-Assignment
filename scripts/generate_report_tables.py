"""Export numeric LaTeX tables/code excerpts only; no report prose is written."""
import _bootstrap

from src.utils import ROOT, atomic_bytes, path, read_json


def tex_escape(text):
    mapping = {'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#',
               '$': r'\$', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in str(text))


def generate_tables(result=None):
    if result is None:
        from summarize_results import summarize
        result = summarize()
    lines = [r'\begin{tabular}{lrrrr}', r'\hline',
             r'Run & Train CE & Val CE & Train acc. & Val acc. \\', r'\hline']
    for row in result['experiments']:
        label = tex_escape(row['experiment'])
        if row['status'] != 'complete':
            lines.append(label + r' & \multicolumn{4}{c}{' + tex_escape(row['status']) + r'} \\')
        else:
            values = [f'{row[k]:.3f}' if 'loss' in k else f'{100 * row[k]:.1f}\\%' for k in ('train_loss', 'val_loss', 'train_accuracy', 'val_accuracy')]
            lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    atomic_bytes(ROOT / 'results/tables/results.tex', ('\n'.join(lines) + '\n').encode())
    stats_file = path('data/statistics.json')
    stats = read_json(stats_file) if stats_file.is_file() else None
    stats_lines = [r'\begin{tabular}{lrr}', r'\hline', r'Channel & Mean & Population std. \\', r'\hline']
    if stats:
        stats_lines += [f'{channel} & {mean:.5f} & {std:.5f}' + r' \\' for channel, mean, std in zip('RGB', stats['mean'], stats['std'])]
    else:
        stats_lines.append(r'\multicolumn{3}{c}{not computed} \\')
    stats_lines += [r'\hline', r'\end{tabular}']
    atomic_bytes(ROOT / 'results/tables/statistics.tex', ('\n'.join(stats_lines) + '\n').encode())
    # This is the actual transform builder, rather than a separately maintained imitation.
    source = (ROOT / 'src/dataset.py').read_text(encoding='utf-8')
    excerpt = source[source.index('def make_transform('):source.index('\n\nclass DogDataset')]
    atomic_bytes(ROOT / 'results/tables/preprocessing_excerpt.py', excerpt.encode())
    print('Updated results/tables/{results,statistics}.tex and preprocessing_excerpt.py; no report document generated')


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    generate_tables()
