#!/usr/bin/env python3
"""Sequential, restartable experiments. Failed runs remain in the record."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/data/full_multigrid"
EXE = ROOT / "cpp/build/rdme_full_multigrid"


def run(name, args, timeout=900):
    DATA.mkdir(exist_ok=True)
    metadata = DATA / f"{name}.json"
    if metadata.exists():
        print(f"saved {name}", flush=True)
        return
    command = [str(EXE), "--cycles", "2", *map(str, args)]
    executable_hash = hashlib.sha256(EXE.read_bytes()).hexdigest()
    print(f"running {name}", flush=True)
    start = time.monotonic()
    with (DATA / f"{name}.csv").open("w") as out, (DATA / f"{name}.log").open("w") as err:
        try:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=timeout)
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    metadata.write_text(json.dumps(dict(command=command, status=status, executable_sha256=executable_hash,
                                       wall_seconds=time.monotonic()-start), indent=2)+"\n")
    print(f"finished {name}: {status}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["accuracy", "many_voxels", "history", "direct", "graph_gap", "block_smoother", "kcycle_schlogl", "baselines", "bistability_block"])
    args = parser.parse_args()
    if args.suite == "accuracy":
        for cap, steps in [(80,20),(96,20),(112,20),(112,40),(112,80)]:
            name=f"transient_k{cap}_s{steps}"
            run(name,["--cap",cap,"--steps",steps,"--shift",2*steps/100,
                "--integrator","cn","--method","mg","--output",DATA/f"{name}_marginal.csv"],timeout=3600)
    elif args.suite == "many_voxels":
        for dimension,cells in [(1,64),(1,128),(1,256),(1,512),(1,1024),
                                (2,4),(2,8),(2,16),(2,32),(3,2),(3,4),(3,8)]:
            for method in ["mg","ilu","kcycle"]:
                # The K-cycle uses the same smoother and hierarchy as the W-cycle; only
                # the coarse solves change to two GCR iterations per level.
                extra = ["--method","mg","--cycle-type","k"] if method=="kcycle" else ["--method",method]
                run(f"voxels_{dimension}d_{cells}_{method}",["--dimension",dimension,"--cells",cells,
                    "--cap",2,"--initial",2,"--linear",1,"--shift",.5,*extra],timeout=600)
    elif args.suite == "history":
        for case,base in [("schlogl",["--cap",80]),
                          ("spatial",["--cells",1024,"--cap",2,"--initial",2,"--linear",1,"--shift",.5])]:
            for key,hierarchy,cycles in [("direct","direct",1),("v","recursive",1),("w","recursive",2)]:
                name=f"history_{case}_{key}"
                run(name+"_solve",[*base,"--hierarchy",hierarchy,"--cycles",cycles,
                    "--history",DATA/f"{name}.csv"],timeout=900)
    elif args.suite == "direct":
        for cap in [20,32,40,64,80]:
            run(f"klu_k{cap}",["--cap",cap,"--method","klu"],timeout=600)
        run("klu_1024_voxels",["--cells",1024,"--cap",2,"--initial",2,"--linear",1,
            "--shift",.5,"--method","klu"],timeout=600)
    elif args.suite == "graph_gap":
        # Fixed voxel graph: test whether the within-pair gap remains useful as
        # the number of retained molecule arrangements grows with the cap.
        base = ["--cells",16,"--initial",2,"--linear",1,"--shift",.5,
                "--method","mg","--max-iterations",300]
        for cap in [2,4,6,8]:
            for label,extra in [("w",["--cycles",2]),
                                ("kcycle",["--cycle-type","k"]),
                                ("direct",["--cycles",1,"--hierarchy","direct"])]:
                name=f"graph_gap_cap{cap}_{label}"
                run(name,[*base,"--cap",cap,"--diffusion",1,*extra,
                    "--history",DATA/f"{name}_history.csv"],timeout=180)
        # Give the direct correction enough iterations to reach the same
        # algebraic tolerance before comparing solve times.
        for cap in [4,6,8]:
            name=f"graph_gap_cap{cap}_direct1200"
            run(name,[*base,"--cap",cap,"--diffusion",1,"--cycles",1,
                "--hierarchy","direct","--max-iterations",1200,
                "--history",DATA/f"{name}_history.csv"],timeout=180)
        # At cap four, vary the diffusion strength while holding the FSP fixed.
        for diffusion in [0.01,0.1,10]:
            for label,extra in [("w",["--cycles",2]),
                                ("direct",["--cycles",1,"--hierarchy","direct"])]:
                name=f"graph_gap_d{diffusion}_{label}"
                run(name,[*base,"--cap",4,"--diffusion",diffusion,*extra,
                    "--history",DATA/f"{name}_history.csv"],timeout=180)
        # Same number of voxels and states, but a different voxel graph.
        for label,extra in [("w",["--cycles",2]),
                            ("direct",["--cycles",1,"--hierarchy","direct"])]:
            name=f"graph_gap_grid2d_{label}"
            run(name,["--dimension",2,"--cells",4,"--cap",4,"--initial",2,
                "--linear",1,"--shift",.5,"--diffusion",1,"--method","mg",
                "--max-iterations",300,*extra,
                "--history",DATA/f"{name}_history.csv"],timeout=180)
        # A bridge between fixed-cap spatial refinement and fixed-graph
        # population refinement: both the grid and cap exceed the smallest
        # settings in the preceding sweeps.
        for cells in [32,64]:
            for label,extra in [("w",["--cycles",2]),
                                ("direct",["--cycles",1,"--hierarchy","direct"])]:
                name=f"graph_gap_bridge_v{cells}_k4_{label}"
                run(name,["--cells",cells,"--cap",4,"--initial",2,"--linear",1,
                    "--shift",.5,"--diffusion",1,"--method","mg",
                    "--max-iterations",300,*extra,
                    "--history",DATA/f"{name}_history.csv"],timeout=180)
    elif args.suite == "block_smoother":
        # Compare population-block smoothing with point smoothing on the same
        # voxel hierarchy, tolerance, cycle type, and sweep count.
        for cells,cap in [(16,4),(16,8),(64,4)]:
            for sweeps in [1,2]:
                for smoother in ["gs","block-ilu"]:
                    name=f"block_v{cells}_k{cap}_{smoother}_s{sweeps}"
                    run(name,["--cells",cells,"--cap",cap,"--initial",2,
                        "--linear",1,"--shift",.5,"--diffusion",1,
                        "--method","mg","--smoother",smoother,
                        "--sweeps",sweeps,"--max-iterations",300,
                        "--history",DATA/f"{name}_history.csv"],timeout=180)
        # Exact sparse block factors give a small-case reference, not a
        # scalable implementation of the population-block idea.
        name="block_v16_k4_exact_s2"
        run(name,["--cells",16,"--cap",4,"--initial",2,"--linear",1,
            "--shift",.5,"--diffusion",1,"--method","mg",
            "--smoother","block","--sweeps",2,
            "--max-iterations",300,"--check",
            "--history",DATA/f"{name}_history.csv"],timeout=180)
        # Nonlinear Schlögl reactions: the same block ordering still applies
        # because each reaction changes the total population by one.
        for cap in [40,60]:
            for smoother in ["gs","block-ilu"]:
                name=f"block_schlogl_v4_k{cap}_{smoother}_s1"
                run(name,["--cells",4,"--cap",cap,"--initial",8,
                    "--diffusion",1,"--shift",.05,"--method","mg",
                    "--smoother",smoother,"--sweeps",1,
                    "--max-iterations",300,
                    "--history",DATA/f"{name}_history.csv"],timeout=180)
            # The same block relaxation without voxel coarse correction tests
            # whether the improvement is genuinely multilevel.
            name=f"block_schlogl_v4_k{cap}_block-ilu_only_s1"
            run(name,["--cells",4,"--cap",cap,"--initial",8,
                "--diffusion",1,"--shift",.05,"--method","gs",
                "--smoother","block-ilu","--sweeps",1,
                "--max-iterations",300,
                "--history",DATA/f"{name}_history.csv"],timeout=180)
    elif args.suite == "kcycle_schlogl":
        # Same settings as the block_smoother Schlögl runs; only the coarse
        # solves change from two plain recursive visits to the K-cycle.
        for cap in [40,60]:
            for smoother in ["gs","block-ilu"]:
                name=f"block_schlogl_v4_k{cap}_{smoother}_s1_kcycle"
                run(name,["--cells",4,"--cap",cap,"--initial",8,
                    "--diffusion",1,"--shift",.05,"--method","mg",
                    "--cycle-type","k","--smoother",smoother,"--sweeps",1,
                    "--max-iterations",300,
                    "--history",DATA/f"{name}_history.csv"],timeout=180)
    elif args.suite == "baselines":
        run("large_voxel_ilu_long",["--cells",1024,"--cap",2,"--initial",2,"--linear",1,
            "--shift",.5,"--method","ilu","--max-iterations",5000],timeout=900)
    elif args.suite == "bistability_block":
        # A standalone W-cycle, not a Krylov preconditioner. The population
        # blocks are relaxed in both directions and voxel levels correct the
        # remaining error. Compare its entire transient with the point-swept
        # W-cycle on the identical truncated RDME.
        for cap in [80,96]:
            name=f"bistability_block_k{cap}_s20"
            run(name,["--cells",4,"--cap",cap,"--initial",8,
                "--diffusion",1,"--shift",.4,"--steps",20,
                "--integrator","cn","--method","mg",
                "--smoother","block-ilu","--sweeps",1,
                "--output",DATA/f"{name}_marginal.csv"],timeout=1800)


if __name__ == "__main__":
    main()
