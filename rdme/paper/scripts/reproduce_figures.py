#!/usr/bin/env python3
"""Rebuild all data figures from the included records and exported transfers."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'paper/scripts'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path,
                        default=ROOT / 'cpp/build/rdme_full_multigrid')
    args = parser.parse_args()
    (ROOT / 'paper/figures').mkdir(parents=True, exist_ok=True)
    for script, options in [
        ('build_full_multigrid_figures.py', []),
        ('show_block_bistability.py', []),
        ('summarize_rare_switching.py', []),
        ('plot_transfer_sparsity.py', ['--executable', str(args.executable.resolve())]),
        ('plot_spatial_front_marginals.py', ['--paper']),
    ]:
        subprocess.run([sys.executable, str(SCRIPTS / script), *options],
                       cwd=ROOT, check=True)
    for ext in ['pdf', 'png']:
        source = ROOT / 'output/figures' / ('rdme_rare_switching.' + ext)
        if source.exists():
            shutil.copy2(source, ROOT / 'paper/figures' / source.name)
    names = ['full_multigrid_convergence', 'full_multigrid_spatial_scaling',
             'block_multigrid_bistability', 'rdme_rare_switching',
             'rdme_transfer_sparsity', 'rdme_transfer_population_block',
             'schlogl_spatial_front_marginals']
    for name in names:
        path = ROOT / 'paper/figures' / (name + '.pdf')
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('Missing figure: ' + str(path))
    print('Rebuilt all seven data figures in ' + str(ROOT / 'paper/figures'))

if __name__ == '__main__':
    main()
