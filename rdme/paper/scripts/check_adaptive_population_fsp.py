#!/usr/bin/env python3
"""Validate adaptive population selection against larger fixed FSPs.

Fixed/adaptive comparisons use identical backward-Euler steps, isolating
truncation from temporal error. A separate uniformization comparison checks
time refinement. Raw commands, executable hashes, and failures are retained.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from itertools import zip_longest

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / 'cpp/build/rdme_full_multigrid'


def vector_error(a, b):
    with a.open() as fa, b.open() as fb:
        error = 0.
        for x, y in zip_longest(csv.DictReader(fa), csv.DictReader(fb)):
            if x is not None and y is not None:
                assert x['state'] == y['state']
            px = float(x['probability']) if x else 0.
            py = float(y['probability']) if y else 0.
            error += abs(px-py)
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
        default=ROOT/'paper/data/adaptive_population_fsp')
    parser.add_argument('--executable', type=Path, default=EXE)
    args = parser.parse_args()
    data = args.output_dir.resolve()
    data.mkdir(parents=True, exist_ok=True)
    exe = args.executable.resolve()
    executable_hash = hashlib.sha256(exe.read_bytes()).hexdigest()

    def run(name, options, expected_success=True, state=True):
        command = [str(exe), '--cycles', '2', '--tolerance', '1e-12',
                   *map(str, options)]
        if state:
            command += ['--state-output', str(data/(name+'_state.csv'))]
        start = time.monotonic()
        with (data/(name+'.csv')).open('w') as out, (data/(name+'.log')).open('w') as err:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=90)
        (data/(name+'.json')).write_text(json.dumps(dict(command=command,
            executable_sha256=executable_hash, status=result.returncode,
            wall_seconds=time.monotonic()-start), indent=2)+'\n')
        assert (result.returncode == 0) == expected_success, (name, (data/(name+'.log')).read_text())
        return list(csv.DictReader((data/(name+'.csv')).open()))

    def validate(rows, steps):
        accepted = [r for r in rows if r['accepted']=='1']
        assert [int(r['step']) for r in accepted] == list(range(1,steps+1))
        previous_cap = -1
        previous_step = 1
        for r in rows:
            assert int(r['cap']) >= previous_cap
            assert int(r['step']) == previous_step
            assert float(r['residual_l1_bound']) <= 1e-12
            if r['accepted']=='1':
                assert float(r['missing_mass'])+2*float(r['algebraic_error_bound']) <= float(r['loss_budget'])
                previous_step += 1
            previous_cap = int(r['cap'])
        return accepted

    results = {}
    base = ['--cells',4,'--initial',0,'--linear',1,'--shift',5,'--steps',10]
    reference = run('linear_fixed', [*base,'--cap',30])
    assert 1-float(reference[-1]['mass']) < 1e-10
    for label,tolerance in [('loose',1e-3),('mid',1e-5),('tight',1e-7)]:
        name='linear_'+label
        rows=run(name,[*base,'--adaptive-fsp','--cap',0,'--fsp-tolerance',tolerance,'--check'])
        accepted=validate(rows,10)
        error=vector_error(data/(name+'_state.csv'),data/'linear_fixed_state.csv')
        assert error <= tolerance+1e-10, (name,error)
        results[name]=dict(final_cap=int(accepted[-1]['cap']),peak_states=int(accepted[-1]['states']),
            reference_states=int(reference[-1]['states']),expansions=len(rows)-len(accepted),
            final_missing_mass=float(accepted[-1]['missing_mass']),reference_l1=error,
            total_seconds=float(accepted[-1]['total_setup_s'])+float(accepted[-1]['total_solve_s']))
        if label=='mid':
            assert any(r['accepted']=='0' and int(r['step'])>1 for r in rows), 'must exercise expansion after an accepted step'
    assert results['linear_tight']['reference_l1'] < results['linear_loose']['reference_l1']

    # A different initial guess must produce the same probability vector
    # within the two runs' truncation budgets.
    rows=run('linear_start8',[*base,'--adaptive-fsp','--cap',8,'--fsp-tolerance',1e-7])
    validate(rows,10)
    assert vector_error(data/'linear_start8_state.csv',data/'linear_tight_state.csv') < 2e-7

    # Pure diffusion never needs population expansion.
    diffusion=['--cells',4,'--initial',2,'--cap',2,'--linear',0,'--shift',5,'--steps',5]
    rows=run('diffusion',[*diffusion,'--adaptive-fsp','--fsp-tolerance',1e-7])
    assert len(rows)==5
    assert all(int(r['cap'])==2 for r in validate(rows,5))

    # Exercise multidimensional transfers and standard V / nonlinear K cycles.
    for label,extra in [('2d',['--dimension',2,'--cells',2]),
                        ('v',['--cells',4,'--cycles',1]),
                        ('k',['--cells',4,'--cycle-type','k'])]:
        rows=run('cycle_'+label,['--initial',0,'--linear',1,'--shift',5,'--steps',2,
            '--adaptive-fsp','--fsp-tolerance',1e-5,'--check',*extra])
        validate(rows,2)

    # Nonlinear reactions, including a separate temporal error check.
    nonlinear=['--cells',2,'--initial',4]
    fixed=run('schlogl_fixed',[*nonlinear,'--cap',48,'--shift',10,'--steps',10])
    rows=run('schlogl_adaptive',[*nonlinear,'--cap',4,'--shift',10,'--steps',10,
        '--adaptive-fsp','--fsp-tolerance',1e-7,'--smoother','block-ilu','--sweeps',1,'--check'])
    accepted=validate(rows,10)
    error=vector_error(data/'schlogl_adaptive_state.csv',data/'schlogl_fixed_state.csv')
    assert error < 1e-7+1e-10,error
    results['schlogl']=dict(final_cap=int(accepted[-1]['cap']),states=int(accepted[-1]['states']),reference_l1=error)
    run('schlogl_uniform',[*nonlinear,'--cap',48,'--shift',1,'--steps',1,'--method','uniformization'])
    rows=run('schlogl_refined',[*nonlinear,'--shift',20,'--steps',20,
        '--adaptive-fsp','--fsp-tolerance',1e-7,'--smoother','block-ilu','--sweeps',1])
    validate(rows,20)
    coarse=vector_error(data/'schlogl_adaptive_state.csv',data/'schlogl_uniform_state.csv')
    fine=vector_error(data/'schlogl_refined_state.csv',data/'schlogl_uniform_state.csv')
    assert fine<coarse,(coarse,fine)
    results['time_refinement']=dict(dt_point1_l1=coarse,dt_point05_l1=fine)

    # Incremental growth must reproduce full rebuilding, including expansion
    # from cap zero, exact/approximate blocks, and the one-voxel bottom solve.
    reuse_cases = [
        ('gs', ['--cells',4,'--smoother','gs']),
        ('block_ilu', ['--cells',4,'--smoother','block-ilu','--sweeps',1]),
        ('block_exact', ['--cells',2,'--smoother','block','--sweeps',1]),
        ('bottom', ['--cells',1]),
        ('2d', ['--cells',2,'--dimension',2,'--smoother','block-ilu','--sweeps',1]),
        ('direct', ['--cells',4,'--hierarchy','direct','--smoother','block-ilu','--sweeps',1]),
        ('k', ['--cells',4,'--cycle-type','k','--smoother','block-ilu','--sweeps',1]),
    ]
    for label,extra in reuse_cases:
        options=['--initial',0,'--cap',0,'--linear',1,'--shift',5,'--steps',10,
                 '--adaptive-fsp','--fsp-tolerance',1e-5,'--check',*extra]
        incremental=run('reuse_'+label,[*options,'--reuse-fsp',1])
        rebuilt=run('rebuild_'+label,[*options,'--reuse-fsp',0])
        validate(incremental,10)
        validate(rebuilt,10)
        for a,b in zip_longest(incremental,rebuilt):
            assert a is not None and b is not None
            for field in ('cap','step','accepted','iterations'):
                assert a[field]==b[field], (label,field,a[field],b[field])
        error=vector_error(data/('reuse_'+label+'_state.csv'),
                           data/('rebuild_'+label+'_state.csv'))
        assert error < 1e-14, (label,error)
        log=(data/('reuse_'+label+'.log')).read_text()
        assert 'FSP reused rows=' in log, label
        if label in ('block_ilu','block_exact','2d','direct','k'):
            assert any(int(line.split('block_factors=')[1].split()[0])>0
                       for line in log.splitlines() if line.startswith('FSP reused')), label
        results['reuse_'+label]=dict(joint_l1_difference=error,attempts=len(incremental))

    # Sparse-storage counts also cover channels without a first-order term.
    for label,rates in [('autocatalytic',(0.01,0,0,0)),
                        ('cubic_death',(0,0.01,0,0)),
                        ('immigration',(0,0,0.1,0)),
                        ('linear_death',(0,0,0,0.1))]:
        options=['--cells',2,'--initial',4,'--cap',6,'--shift',10,'--check']
        for flag,value in zip(('--c1','--c2','--c3','--c4'),rates):
            options += [flag,value]
        run('storage_'+label,options,state=False)

    # Fail explicitly instead of accepting an unverified step.
    for label,options in [('bad_tolerance',['--fsp-tolerance','nan']),
                          ('unsupported_cn',['--integrator','cn']),
                          ('failed_solve',['--max-iterations',1,'--initial',0,'--cap',4,'--linear',1]),
                          ('resource_guard',['--cap',10000])]:
        run(label,['--adaptive-fsp',*options],expected_success=False,state=False)

    (data/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
