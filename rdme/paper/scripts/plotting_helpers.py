#!/usr/bin/env python3
"""Rebuild publication figures from recorded CSV data using gnuplot.

Uses only the Python standard library and gnuplot (pdfcairo/pngcairo).
Run from any directory. No values in the plots are fitted or interpolated.
"""
import csv
import re
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data"
FIG = PAPER / "figures"


def read(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def block(name, rows):
    return f"${name} << EOD\n" + "\n".join(" ".join(map(str, r)) for r in rows) + "\nEOD\n"


def render(name, definitions, commands, height=3.0, monochrome=False):
    FIG.mkdir(parents=True, exist_ok=True)
    common = """set encoding utf8
set border 3 linewidth 0.7
set tics nomirror out scale 0.6
set grid ytics lc rgb '#e6e6e6' lw 0.5
set style line 1 lc rgb '#194f78' lw 1.8 pt 7 ps 0.65 dt 1
set style line 2 lc rgb '#b95223' lw 1.8 pt 5 ps 0.65 dt 2
set style line 3 lc rgb '#357257' lw 1.8 pt 9 ps 0.75 dt 3
set style line 4 lc rgb '#555555' lw 1.4 pt 11 ps 0.65 dt 4
set key font ',8' spacing 1.1 samplen 2
set xtics font ',9'
set ytics font ',9'
set xlabel offset 0,0.3
set ylabel offset 1,0
""".replace("\\+", "")
    if monochrome:
        # Encode series with dash patterns and marker shapes, without color or shading.
        common = re.sub(r"lc rgb '#[0-9a-fA-F]{6}'", "lc rgb '#000000'", common)
        common = re.sub(r"set grid[^\n]*", "unset grid", common)
        commands = re.sub(r"lc rgb '#[0-9a-fA-F]{6}'", "lc rgb '#000000'", commands)
    for ext in ("pdf", "png"):
        terminal = (f"pdfcairo enhanced color font 'Times,10' size 6.8in,{height}in"
                    if ext == "pdf" else
                    f"pngcairo enhanced font 'Times,18' size 1600,{int(1600*height/6.8)}")
        script = f"set terminal {terminal}\nset output '{(FIG / (name+'.'+ext)).as_posix()}'\n"
        script += common + definitions + commands + "\nunset output\n"
        subprocess.run(["gnuplot"], input=script, text=True, check=True)
