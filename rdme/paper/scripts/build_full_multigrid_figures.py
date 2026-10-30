#!/usr/bin/env python3
"""Figures for the full-RDME multigrid cycle. Uses recorded results only."""
import csv
import json
from pathlib import Path
from functools import partial
from plotting_helpers import block, render as _render

render = partial(_render, monochrome=True)

PAPER=Path(__file__).resolve().parents[1]
DATA=PAPER/"data/full_multigrid"


def read(name):
    with (DATA/name).open() as stream:
        return list(csv.DictReader(stream))


def last(name):
    rows=read(name)
    if not rows:
        raise ValueError(f"Incomplete run: {name}")
    return rows[-1]


def terminal(cap,steps):
    rows=read(f"transient_k{cap}_s{steps}_marginal.csv")
    selected=[r for r in rows if abs(float(r["time"])-100)<1e-10]
    assert len(selected)==cap+1, (cap,steps,"incomplete transient")
    return [float(r["probability"]) for r in selected]


def difference(a,b):
    return sum(abs((a[i] if i<len(a) else 0)-(b[i] if i<len(b) else 0))
               for i in range(max(len(a),len(b))))


def refinement():
    result=[]
    for cap,steps in [(80,20),(96,20),(112,20),(112,40),(112,80)]:
        values=terminal(cap,steps)
        row=last(f"transient_k{cap}_s{steps}.csv")
        assert row["converged"]=="1"
        assert abs(float(row["mass"])-sum(values))<1e-9
        result.append(dict(cap=cap,steps=steps,dt=100/steps,states=int(row["states"]),
          mass=sum(values),high_mass=sum(values[21:]),
          min_probability=float(row["min_probability"]),
          modes=[i for i in range(1,len(values)-1) if values[i]>values[i-1] and values[i]>values[i+1]],
          difference_from_previous=None if not result else difference(values,terminal(result[-1]["cap"],result[-1]["steps"]))))
    (DATA/"refinement_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def convergence():
    definitions=""
    plots="set multiplot layout 1,2 margins 0.10,0.98,0.20,0.88 spacing 0.13,0\n"
    for case,title in [("schlogl","Schlogl: 1,929,501 states"),("spatial","1D: 1024 voxels, population cap 2")]:
        curves=[]
        for j,(method,label) in enumerate([("direct","Direct CME correction"),("v","V-cycle"),("w","W-cycle")],1):
            name=f"history_{case}_{method}.csv"
            rows=read(name)
            definitions+=block(f"{case}{method}",[(r["iteration"],r["residual_l1_bound"]) for r in rows])
            curves.append(f"${case}{method} u 1:2 w l ls {j} t '{label}'")
        plots+=f"set title '{title}'\nset xlabel 'Cycles'\nset ylabel 'Residual bound on solution error'\nset logscale y\nset format y '10^{{%L}}'\nset yrange [1e-10:*]\nset xrange [0:*]\nset key top right font ',10'\nplot "+", ".join(curves)+"\n"
    render("full_multigrid_convergence",definitions,plots+"unset multiplot\n")


def spatial(include_kcycle=True):
    definitions=""
    for d,cells in [(1,[64,128,256,512,1024]),(2,[4,8,16,32]),(3,[2,4,8])]:
        for method in (["mg","kcycle","ilu"] if include_kcycle else ["mg","ilu"]):
            rows=[last(f"voxels_{d}d_{c}_{method}.csv") for c in cells]
            definitions+=block(f"d{d}{method}",[(int(r["cells"])**d,r["iterations"],r["converged"],
                       float(r["setup_s"])+float(r["solve_s"])) for r in rows])
    plots="set multiplot layout 1,3 margins 0.08,0.98,0.22,0.86 spacing 0.10,0\n"
    for d in [1,2,3]:
        lo,hi,ticks={1:(45,1500,"64,128,256,512,1024"),2:(12,1500,"16,64,256,1024"),3:(6,700,"8,64,512")}[d]
        plots+=f"""set title '{d}D voxel grid'
set logscale xy
set xlabel 'Voxels'
{'set' if d==1 else 'unset'} ylabel {'"Cycles or GMRES iterations"' if d==1 else ''}
set yrange [5:1800]
set xrange [{lo}:{hi}]
set format y '%g'
set xtics ({ticks}) rotate by -30
{'set key bottom right font ",9"' if d==1 else 'unset key'}
plot $d{d}mg u 1:2 w lp ls 1 t 'W-cycle', \
     $d{d}kcycle u 1:2 w lp ls 3 t 'K-cycle', \
     $d{d}ilu u 1:($3==1?$2:1/0) w lp ls 2 t 'ILU-GMRES', \
     $d{d}ilu u 1:($3==0?$2:1/0) w p pt 6 lc rgb '#b95223' t 'Iteration limit'
"""
    if not include_kcycle:
        for d in [1,2,3]:
            plots = plots.replace(f"$d{d}kcycle u 1:2 w lp ls 3 t 'K-cycle',", "")
    name = "full_multigrid_spatial_scaling" if include_kcycle else "full_multigrid_spatial_w_scaling"
    render(name,definitions,plots+"unset multiplot\n",height=2.9)










if __name__=="__main__":
    summary=refinement()
    convergence()
    spatial()
    spatial(include_kcycle=False)
