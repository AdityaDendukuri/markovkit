#!/usr/bin/env python3
"""Check and plot bistability from standalone, block-smoothed RDME W-cycles."""
import csv
import json
from pathlib import Path

from functools import partial
from plotting_helpers import block, render as _render

render = partial(_render, monochrome=True)

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data/full_multigrid"


def read(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def marginal(name, time):
    rows = [r for r in read(name) if abs(float(r["time"]) - time) < 1e-10]
    return [float(r["probability"]) for r in rows]


def modes(values):
    return [(i, values[i]) for i in range(1, len(values)-1)
            if values[i] > values[i-1] and values[i] > values[i+1]]


def main():
    block96 = read("bistability_block_k96_s20_marginal.csv")
    block80 = marginal("bistability_block_k80_s20_marginal.csv", 100)
    point96 = marginal("transient_k96_s20_marginal.csv", 100)
    point112 = marginal("transient_k112_s20_marginal.csv", 100)
    end96 = marginal("bistability_block_k96_s20_marginal.csv", 100)
    assert len(block80) == 81 and len(end96) == 97 and len(point112) == 113
    point_error = sum(abs(a-b) for a,b in zip(end96, point96))
    cap_error = sum(abs((end96[i] if i<len(end96) else 0)-point112[i])
                    for i in range(len(point112)))
    mass = sum(end96)
    assert point_error < 1e-6 and mass > .9999
    result = dict(states=3921225, modes=modes(end96), mass=mass,
                  high_basin_mass=sum(end96[21:]),
                  same_cap_point_vs_block_l1=point_error,
                  cap96_vs_cap112_l1=cap_error)
    (DATA / "bistability_block_summary.json").write_text(json.dumps(result,indent=2)+"\n")

    definitions = ""
    for time in [25,50,100]:
        definitions += block(f"time{time}",
            [(r["population"],r["probability"]) for r in block96
             if abs(float(r["time"])-time)<1e-10])
    by_time = {}
    for row in block96:
        by_time.setdefault(float(row["time"]),[]).append(float(row["probability"]))
    definitions += block("high", [(t,sum(values[21:])) for t,values in by_time.items()])
    for cap,values in [(80,block80),(96,end96),(112,point112)]:
        definitions += block(f"cap{cap}",list(enumerate(values)))
    for smoother,name in [("point","transient_k96_s20.csv"),
                          ("block","bistability_block_k96_s20.csv")]:
        definitions += block(smoother,[(2.5 if int(r["step"])==1 else
                                       5*(int(r["step"])-1),r["iterations"])
                                      for r in read(name)])
    render("block_multigrid_bistability",definitions,"""set multiplot layout 2,2 margins 0.09,0.98,0.11,0.93 spacing 0.12,0.18
set title '(a) Two population peaks emerge'
set xlabel 'Total molecule count, k'
set ylabel 'Probability'
set xrange [0:80]
set yrange [0:0.08]
set key top right
plot $time25 u 1:2 w l ls 4 t 't = 25', $time50 u 1:2 w l ls 1 t 't = 50', $time100 u 1:2 w l ls 2 t 't = 100'
set title '(b) High-population probability'
set xlabel 'Time'
set ylabel 'P(k >= 21)'
set xrange [0:100]
set yrange [0:0.5]
unset key
plot $high u 1:2 w lp ls 2
set title '(c) Population-cap check at t = 100'
set xlabel 'Total molecule count, k'
set ylabel 'Probability'
set xrange [0:80]
set yrange [0:0.05]
set key top right
plot $cap80 u 1:2 w l ls 4 t 'cap 80, block W', $cap96 u 1:2 w l ls 2 t 'cap 96, block W', $cap112 u 1:2 w l ls 1 t 'cap 112, point W'
set title '(d) W-cycles per solve, cap 96'
set xlabel 'Time'
set ylabel 'Cycles per linear solve'
set xrange [0:100]
set yrange [0:*]
set key top left
plot $point u 1:2 w lp ls 1 t 'Point Gauss-Seidel', $block u 1:2 w lp ls 2 t 'Block Gauss-Seidel'
unset multiplot
""",height=5.4)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
