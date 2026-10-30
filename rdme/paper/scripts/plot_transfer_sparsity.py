#!/usr/bin/env python3
"""Export the solver's stored transfers and plot their exact supports.

Standard library + gnuplot only. Uses the four-voxel, cap-40 hierarchy from
Table 3's Schlogl relaxation comparison. Full plots retain solver order;
the block comparison also shows an explicitly recorded permutation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
DATA = PAPER / 'data/transfer_sparsity'
FIG = PAPER / 'figures'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_level(prefix, meta):
    path = Path(f'{prefix}_level{meta["level"]}.csv')
    with path.open() as stream:
        records = list(csv.DictReader(stream))
    nf, nc = int(meta['fine_states']), int(meta['coarse_states'])
    assert len(records) == nf == int(meta['nnz'])
    sums = [[] for _ in range(nc)]
    coarse_population = [None] * nc
    fine_population = []
    for i, row in enumerate(records):
        assert int(row['fine_state']) == i
        j, k, w = int(row['coarse_state']), int(row['population']), float(row['weight'])
        assert 0 <= j < nc and math.isfinite(w) and w > 0
        assert coarse_population[j] in (None, k), 'Transfer crosses population blocks'
        coarse_population[j] = k
        fine_population.append(k)
        sums[j].append(w)
    assert all(x is not None for x in coarse_population)
    error = max(abs(math.fsum(values) - 1) for values in sums)
    assert error < 1e-12, error
    return path, records, fine_population, coarse_population, error


def panel(path, nf, nc, restriction, title):
    # Source uses zero-based solver indices. Reverse the row axis as in a spy plot.
    cols, rows = (nf, nc) if restriction else (nc, nf)
    using = '1:2' if restriction else '2:1'
    return f'''set title {json.dumps(title)}
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:{cols - .5}]
set yrange [{rows - .5}:-0.5]
set xtics 0,{max(1, (cols-1)//2)},{cols-1}
set ytics 0,{max(1, (rows-1)//2)},{rows-1}
plot {json.dumps(str(path))} every ::1 using {using} with dots lc rgb 'black' notitle
'''


def render(stem, commands, height):
    for extension in ('pdf', 'png'):
        out = FIG / f'{stem}.{extension}'
        terminal = (f"pdfcairo enhanced monochrome font 'Times,10' size 6.8in,{height}in"
                    if extension == 'pdf' else
                    f"pngcairo enhanced monochrome font 'Times,17' size 1800,{int(1800*height/6.8)}")
        script = f'''set terminal {terminal}
set output {json.dumps(str(out))}
set datafile separator ','
set border 15 lw 0.5
set tics nomirror out scale 0.4
set format x '%g'
set format y '%g'
set xtics font ',8'
set ytics font ',8'
unset key
{commands}
unset output
'''
        (DATA / f'{stem}_{extension}.gnuplot').write_text(script)
        subprocess.run(['gnuplot'], input=script, text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=ROOT/'cpp/build/rdme_full_multigrid')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    # Match a recorded paper experiment, rather than inventing a small toy space.
    source = PAPER/'data/full_multigrid/block_schlogl_v4_k40_block-ilu_s1.csv'
    with source.open() as stream:
        experiment = next(csv.DictReader(stream))
    prefix = DATA/'schlogl_k40'
    command = [str(args.executable.resolve()), '--cells', experiment['cells'],
               '--dimension', experiment['dimension'], '--cap', experiment['cap'],
               '--initial', '8', '--cycles', '2', '--smoother', 'gs', '--check',
               '--transfer-output', str(prefix), '--export-only']
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    (DATA/'schlogl_k40_export.log').write_text(result.stdout + result.stderr)
    with Path(f'{prefix}_levels.csv').open() as stream:
        metadata = list(csv.DictReader(stream))
    assert len(metadata) == 2
    assert int(metadata[0]['fine_states']) == int(experiment['states'])
    commands = 'set multiplot layout 2,2 margins 0.12,0.98,0.12,0.91 spacing 0.16,0.18\n'
    levels = []
    checks = []
    for i, meta in enumerate(metadata):
        path, records, fp, cp, error = load_level(prefix, meta)
        levels.append((meta, path, records, fp, cp))
        nf, nc = int(meta['fine_states']), int(meta['coarse_states'])
        for restriction, name in [(True, 'R'), (False, 'P')]:
            letter = 'abcd'[2*i + (not restriction)]
            rows, cols = (nc, nf) if restriction else (nf, nc)
            title = f'({letter}) {name}: {meta["fine_voxels"]} to {meta["coarse_voxels"]} voxels ({rows:,} x {cols:,})'
            commands += panel(path, nf, nc, restriction, title)
        checks.append(dict(level=i, fine_states=nf, coarse_states=nc, nnz=nf,
                           max_column_sum_error=error, csv_sha256=sha256(path)))
    render('rdme_transfer_sparsity', commands + 'unset multiplot\n', 4.8)

    # A full population block from that same export shows the individual entries.
    meta, path, records, fp, cp = levels[0]
    fine_ids = [i for i, k in enumerate(fp) if k == 4]
    coarse_ids = [j for j, k in enumerate(cp) if k == 4]
    block_path = DATA/'schlogl_k40_population4.csv'
    with block_path.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fine_state', 'coarse_state', 'population', 'weight'])
        for i in fine_ids:
            row = records[i]
            writer.writerow([i-fine_ids[0], int(row['coarse_state'])-coarse_ids[0], 4, row['weight']])
    assert len(fine_ids) == 35 and len(coarse_ids) == 5
    grouped_path = DATA/'schlogl_k40_population4_grouped.csv'
    permutation = sorted(fine_ids, key=lambda i: (int(records[i]['coarse_state']), i))
    assert sorted(permutation) == fine_ids
    with grouped_path.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fine_state', 'coarse_state', 'population', 'weight', 'original_fine_state'])
        for new_i, old_i in enumerate(permutation):
            row = records[old_i]
            writer.writerow([new_i, int(row['coarse_state'])-coarse_ids[0],
                             4, row['weight'], old_i])
    counts = [sum(int(records[i]['coarse_state']) == j for i in fine_ids)
              for j in coarse_ids]
    assert counts == [5, 8, 9, 8, 5]
    commands = 'set multiplot layout 2,2 margins 0.10,0.98,0.12,0.91 spacing 0.14,0.18\n'
    for grouped in (False, True):
        path_to_plot = grouped_path if grouped else block_path
        for restriction, name in [(True, 'R_4'), (False, 'P_4')]:
            letter = 'abcd'[2*grouped + (not restriction)]
            order = ('fine states grouped by coarse counts' if grouped else 'solver ordering')
            commands += 'unset object\n'
            if grouped:
                start = 0
                for j, count in enumerate(counts):
                    x1, x2, y1, y2 = start-.5, start+count-.5, j-.5, j+.5
                    if not restriction:
                        x1, x2, y1, y2 = y1, y2, x1, x2
                    commands += (f"set object {j+1} rect from {x1},{y1} to {x2},{y2} "
                                 "front fs empty border lc rgb 'black' lw 0.5\n")
                    start += count
            commands += panel(path_to_plot, 35, 5, restriction,
                              f'({letter}) {name}: {order}').replace(
                "with dots lc rgb 'black'", "with points pt 7 ps 0.4 lc rgb 'black'")
    render('rdme_transfer_population_block', commands + 'unset multiplot\n', 4.8)
    (DATA/'provenance.json').write_text(json.dumps(dict(
        experiment=str(source.relative_to(ROOT)), experiment_sha256=sha256(source),
        executable_sha256=sha256(args.executable),
        source_sha256=sha256(ROOT/'cpp/rdme_full_multigrid.cpp'), command=command,
        ordering='Increasing total population, then solver lexicographic weak-composition rank; zero-based indices',
        sampled=False, checks=checks,
        block_comparison=dict(population=4, group_sizes=counts,
            permutation=permutation, grouped_csv_sha256=sha256(grouped_path)),
    ), indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
