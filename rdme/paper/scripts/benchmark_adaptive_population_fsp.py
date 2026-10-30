#!/usr/bin/env python3
"""Large, sequential adaptive/fixed Schlögl comparison at identical BE steps.

No cascading smoothers are used: state selection and multigrid smoothing
are deliberately separate questions. Do not compare these BE timings with
the older Crank--Nicolson timings as if their temporal errors were equal.
"""
import argparse
import csv
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT/'cpp/build/rdme_full_multigrid'
DATA = ROOT/'paper/data/adaptive_population_fsp/large_schlogl'


def run(name, options, timeout):
    DATA.mkdir(parents=True, exist_ok=True)
    meta = DATA/(name+'.json')
    command = [str(EXE), '--cells','4','--initial','8','--diffusion','1',
        '--integrator','be','--shift','.2','--steps','20','--method','mg',
        '--cycles','2','--smoother','block-ilu','--sweeps','1',
        '--tolerance','1e-10',*map(str,options),
        '--output',str(DATA/(name+'_marginal.csv')),
        '--state-output',str(DATA/(name+'_state.csv'))]
    digest = hashlib.sha256(EXE.read_bytes()).hexdigest()
    if meta.exists():
        saved = json.loads(meta.read_text())
        if saved['command']==command and saved['executable_sha256']==digest:
            print('saved',name,saved['status'],flush=True)
            return saved['status']
        raise RuntimeError('existing record differs; preserve it before a new run: '+str(meta))
    print('running',name,flush=True)
    start = time.monotonic()
    status = 'interrupted'
    try:
        with (DATA/(name+'.csv')).open('w') as out, (DATA/(name+'.log')).open('w') as err:
            try:
                status = subprocess.run(command,stdout=out,stderr=err,timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                status = 'timeout'
    finally:
        meta.write_text(json.dumps(dict(command=command,executable_sha256=digest,
            status=status,wall_seconds=time.monotonic()-start),indent=2)+'\n')
    print('finished',name,status,flush=True)
    return status


def summary():
    def rows(name):
        return list(csv.DictReader((DATA/(name+'.csv')).open()))
    a=rows('adaptive_eps1e5')
    accepted=[r for r in a if r['accepted']=='1']
    fixed=rows('fixed_k112')
    assert len(accepted)==len(fixed)==20
    assert [int(r['step']) for r in accepted]==list(range(1,21))
    for r in accepted:
        assert float(r['missing_mass'])+2*float(r['algebraic_error_bound'])<=float(r['loss_budget'])
    error=0.
    with (DATA/'adaptive_eps1e5_state.csv').open() as fa, (DATA/'fixed_k112_state.csv').open() as fb:
        for x,y in zip_longest(csv.DictReader(fa),csv.DictReader(fb)):
            if x and y:assert x['state']==y['state']
            error+=abs((float(x['probability']) if x else 0.)-(float(y['probability']) if y else 0.))
    last=accepted[-1]
    fixed_s=float(fixed[0]['setup_s'])+sum(float(r['solve_s']) for r in fixed)
    adaptive_s=float(last['total_setup_s'])+float(last['total_solve_s'])
    adaptive_meta=json.loads((DATA/'adaptive_eps1e5.json').read_text())
    fixed_meta=json.loads((DATA/'fixed_k112.json').read_text())
    with (DATA/'adaptive_eps1e5_marginal.csv').open() as stream:
        final_marginal=[r for r in csv.DictReader(stream) if float(r['time'])==100.]
    high=sum(float(r['probability']) for r in final_marginal if int(r['population'])>=21)
    adaptive_peak=max(int(r['peak_rss_bytes']) for r in a)
    fixed_peak=max(int(r['peak_rss_bytes']) for r in fixed)
    # Preserve anomalous component timings rather than correcting raw records.
    # Compare the separately measured monotonic process elapsed times.
    component_consistent=(fixed_s<=fixed_meta['wall_seconds']+1 and
                          adaptive_s<=adaptive_meta['wall_seconds']+1)
    elapsed_speedup=fixed_meta['wall_seconds']/adaptive_meta['wall_seconds']
    record=dict(model='four-voxel Schlogl, initial (2,2,2,2), D=1, T=100',
        integrator='backward Euler',dt=5.,steps=20,
        smoother='population-block ILU(0), one forward and reverse sweep',cycle='W',
        adaptive=dict(final_cap=int(last['cap']),peak_states=max(int(r['states']) for r in a),
            accepted_state_steps=sum(int(r['states']) for r in accepted),
            attempted_state_steps=sum(int(r['states']) for r in a),
            expansions=len(a)-len(accepted),caps=sorted({int(r['cap']) for r in a}),
            cycles=sum(int(r['iterations']) for r in a),
            rejected_cycles=sum(int(r['iterations']) for r in a if r['accepted']=='0'),
            missing_mass=float(last['missing_mass']),algebraic_error_bound=float(last['algebraic_error_bound']),
            high_population_probability=high,
            setup_s=float(last['total_setup_s']),solve_s=float(last['total_solve_s']),total_s=adaptive_s,
            wall_s=adaptive_meta['wall_seconds'],
            peak_rss_bytes=adaptive_peak),
        fixed=dict(cap=112,states=int(fixed[0]['states']),missing_mass=1-float(fixed[-1]['mass']),
            setup_s=float(fixed[0]['setup_s']),solve_s=sum(float(r['solve_s']) for r in fixed),
            total_s=fixed_s,wall_s=fixed_meta['wall_seconds'],
            high_population_probability=float(fixed[-1]['high_mass']),
            cycles=sum(int(r['iterations']) for r in fixed),
            peak_rss_bytes=fixed_peak),
        joint_l1_difference=error,speedup=elapsed_speedup,
        speedup_basis='external monotonic process elapsed time, including state output',
        component_timing_consistent=component_consistent,
        component_timing_note='Fixed step 15 reports 94.42 s versus roughly 30 s for adjacent steps; summed internal time exceeds external elapsed time. Internal component totals are preserved but not used for speedup.',
        retained_state_reduction=1-int(last['states'])/int(fixed[0]['states']),
        peak_memory_reduction=1-adaptive_peak/fixed_peak,
        temporal_error='not included in FSP tolerance; both runs use identical BE steps',
        rob_saye_scope='future smoother design and convergence analysis; not used in these runs')
    assert error<=1e-5+sum(float(r['residual_l1_bound']) for r in fixed)+1e-9,error
    (DATA/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    report=f'''# Adaptive FSP on the large Schlogl transient

Both runs solve the same four-voxel model with initial state `(2,2,2,2)`,
`D=1`, and final time 100. Both use 20 backward-Euler steps of length 5,
W-cycles, population-block ILU(0) relaxation with one sweep in each
direction, and per-solve algebraic tolerance `1e-10`.

Adaptive FSP starts at cap 8 and requests cumulative loss tolerance `1e-5`.
The fixed cap-112 run is an accurate, conservative reference, not a claim
that cap 112 is the smallest fixed cap that would satisfy this tolerance.
Timings are single serial runs on the same machine.

| Quantity | Adaptive FSP | Fixed FSP |
|---|---:|---:|
| Final cap | {int(last['cap'])} | 112 |
| Maximum retained states | {int(last['states']):,} | {int(fixed[0]['states']):,} |
| External elapsed time, including state-file output (s) | {adaptive_meta['wall_seconds']:.2f} | {fixed_meta['wall_seconds']:.2f} |
| Peak process memory (GiB) | {adaptive_peak/2**30:.3f} | {fixed_peak/2**30:.3f} |
| Missing mass | {float(last['missing_mass']):.6g} | {1-float(fixed[-1]['mass']):.6g} |
| Total cycles, including rejected attempts | {record['adaptive']['cycles']} | {record['fixed']['cycles']} |

Adaptive/fixed full-state l1 difference: **{error:.6g}**.
Adaptive high-population probability (`N >= 21`): {high:.12g};
fixed: {float(fixed[-1]['high_mass']):.12g}.

The adaptive run makes {record['adaptive']['expansions']} expansions. Its
timings include rejected solves and hierarchy rebuilds. It retains
{100*record['retained_state_reduction']:.1f}% fewer states at its largest
cap and uses {100*record['peak_memory_reduction']:.1f}% less peak process
memory than this reference. The measured external elapsed-time speedup is
{record['speedup']:.3f}.

## Meaning and limits

The controller selects a population range from an accuracy budget and
works on a multimillion-state nonlinear problem. Every accepted interval
satisfies its cumulative discrete loss budget, including the accumulated
linear-solve error allowance. All arrangements of each retained population
are still stored. Adaptation does not remove combinatorial state growth.

Both runs have the same backward-Euler temporal discretization. The loss
tolerance excludes continuous-time error. These timings must not be
compared directly with the established Crank--Nicolson timings as if their
time-discretization errors were equal. The comparison does not establish
that all competing RDME solvers are intractable.

The fixed run has an internal timing anomaly: step 15 reports 94.42 seconds
while neighboring steps report about 30 seconds. Its summed internal times
exceed the separately measured external elapsed time. Raw component timings
are preserved in the CSV and JSON, but the runtime comparison above uses
only external monotonic process elapsed time. These are single-run timings;
the anomaly prevents a reliable component-by-component timing comparison.

## Rob Saye's prospective contribution

This benchmark uses the existing population-block smoother, with no
cascading smoother. Adaptive population selection, the voxel hierarchy,
and the existing graph-energy approximation analysis are the current work.
Rob's potential contribution is a separate improvement to relaxation and
the convergence analysis: reduce slowly damped error at low setup and
storage cost while keeping state selection and transfers fixed.

Raw commands, executable hashes, rejected attempts, marginal distributions,
and final state vectors are saved in
`paper/data/adaptive_population_fsp/large_schlogl`.
Reproduce with `python3 paper/scripts/benchmark_adaptive_population_fsp.py all`.
'''
    (ROOT/'paper/ADAPTIVE_FSP_SCALABILITY.md').write_text(report)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('case',choices=['adaptive','fixed','all','summary'])
    parser.add_argument('--timeout',type=int,default=2400)
    args=parser.parse_args()
    status=0
    if args.case in ['adaptive','all']:
        status=run('adaptive_eps1e5',['--adaptive-fsp','--cap','8','--fsp-tolerance','1e-5'],args.timeout)
    if status==0 and args.case in ['fixed','all']:
        status=run('fixed_k112',['--cap','112'],args.timeout)
    if status==0 and args.case in ['all','summary']:summary()
    if status!=0:raise SystemExit(1)
