#!/usr/bin/env python3
"""Sequential generic/fused ILU comparisons with CPU time and full FSP vectors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,default=ROOT/'cpp/build/rdme_full_multigrid')
    parser.add_argument('--baseline-executable',type=Path,
                        help='Optional saved implementation before the structural changes')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'paper/data/structured_population_blocks')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--case',choices=['rare','adaptive','all'],default='all')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--reflection',action='store_true',
                        help='Also enable exact reflection reduction in the optimized variant')
    args=parser.parse_args()
    data=args.output_dir.resolve()
    data.mkdir(parents=True,exist_ok=True)
    exe=args.executable.resolve()
    cases={
        'rare':['--cells',4,'--initial',0,'--cap',32 if args.quick else 56,
                '--integrator','cn','--shift',.2,'--steps',5 if args.quick else 50,
                '--c1',.0258,'--c2',.0024,'--c3',.00138,'--c4',.0829,'--tolerance','1e-13'],
        'adaptive':['--cells',4,'--initial',8,'--cap',8,'--adaptive-fsp',
                    '--fsp-tolerance','1e-5','--shift',.5,'--steps',2 if args.quick else 10,
                    '--tolerance','1e-10'],
    }
    summary=json.loads((data/'summary.json').read_text()) if (data/'summary.json').exists() else {}
    def run(case,kernel,repeat):
        prefix=data/f'{case}_{kernel}_{repeat}'
        chosen_exe=args.baseline_executable.resolve() if kernel=='reference' and args.baseline_executable else exe
        chosen_digest=hashlib.sha256(chosen_exe.read_bytes()).hexdigest()
        selectors=[] if chosen_exe!=exe else ['--block-kernel',kernel,'--profile']
        if kernel=='structured' and args.reflection:selectors+=['--reflection','1']
        command=[str(chosen_exe),'--smoother','block-ilu','--cycles','2','--sweeps','1',
                 *selectors,*map(str,cases[case]),
                 '--state-output',str(prefix)+'_state.csv']
        meta=prefix.with_suffix('.json')
        if meta.exists():
            saved=json.loads(meta.read_text())
            assert saved['command']==command and saved['executable_sha256']==chosen_digest
            assert saved['status']==0
            return saved
        print('running',prefix.name,flush=True)
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
        start=time.monotonic()
        status='interrupted'
        try:
            with prefix.with_suffix('.csv').open('w') as out,prefix.with_suffix('.log').open('w') as err:
                status=subprocess.run(command,stdout=out,stderr=err,timeout=1800).returncode
        finally:
            usage=resource.getrusage(resource.RUSAGE_CHILDREN)
            record=dict(command=command,executable_sha256=chosen_digest,status=status,
                        wall_seconds=time.monotonic()-start,
                        cpu_user_seconds=usage.ru_utime-before.ru_utime,
                        cpu_system_seconds=usage.ru_stime-before.ru_stime)
            meta.write_text(json.dumps(record,indent=2)+'\n')
        assert status==0,(case,kernel,prefix.with_suffix('.log').read_text())
        with prefix.with_suffix('.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        record.update(setup_seconds=float(rows[-1]['total_setup_s']) if case=='adaptive' else float(rows[0]['setup_s']),
                      solve_seconds=sum(float(r['solve_s']) for r in rows),
                      peak_rss_bytes=max(int(r['peak_rss_bytes']) for r in rows),
                      storage_bytes=int(rows[-1].get('storage_bytes',0)),
                      cycles=sum(int(r['iterations']) for r in rows),
                      full_states=int(rows[-1].get('full_states',rows[-1]['states'])),
                      final_states=int(rows[-1]['states']))
        meta.write_text(json.dumps(record,indent=2)+'\n')
        print('finished',prefix.name,round(record['wall_seconds'],3),flush=True)
        return record
    for case in cases:
        if args.case not in ['all',case]:continue
        summary[case]=[]
        for repeat in range(args.repeats):
            pair={}
            for kernel in (['reference','structured'] if repeat%2==0 else ['structured','reference']):
                pair[kernel]=run(case,kernel,repeat)
            difference=0.; peak={}; minimum={}
            marginals={}
            for kernel in ['reference','structured']:
                # Contiguous population blocks have C(k+3,3) states for four voxels.
                import math
                k=0; end=1; marginal=[]; total=0.; min_value=1.
                with (data/f'{case}_{kernel}_{repeat}_state.csv').open() as stream:
                    for i,row in enumerate(csv.DictReader(stream)):
                        while i>=end:
                            marginal.append(total); total=0.; k+=1
                            end+=math.comb(k+3,3)
                        value=float(row['probability']);total+=value;min_value=min(min_value,value)
                    marginal.append(total)
                marginals[kernel]=marginal
                peak[kernel]=marginal[20] if len(marginal)>20 else 0.
                minimum[kernel]=min_value
            with (data/f'{case}_reference_{repeat}_state.csv').open() as a, (data/f'{case}_structured_{repeat}_state.csv').open() as b:
                left=csv.DictReader(a);right=csv.DictReader(b)
                count=0
                for x,y in zip(left,right,strict=True):
                    assert x['state']==y['state']
                    difference+=abs(float(x['probability'])-float(y['probability']));count+=1
            assert count==pair['reference']['full_states']==pair['structured']['full_states']
            assert difference<1e-10,(case,difference)
            peak_relative=abs(peak['reference']-peak['structured'])/max(1e-300,abs(peak['reference']))
            assert peak_relative<1e-6,(case,peak_relative)
            component_consistent=all(pair[k]['setup_seconds']+pair[k]['solve_seconds']
                                     <=pair[k]['wall_seconds']+1. for k in ['reference','structured'])
            pair.update(joint_l1_difference=difference,peak20=peak,peak20_relative_difference=peak_relative,
                        min_probability=minimum,
                        component_timing_consistent=component_consistent,
                        wall_speedup=pair['reference']['wall_seconds']/pair['structured']['wall_seconds'],
                        solve_speedup=pair['reference']['solve_seconds']/pair['structured']['solve_seconds']
                                      if component_consistent else None)
            summary[case].append(pair)
            (data/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({case:[{k:p[k] for k in ['joint_l1_difference','peak20_relative_difference','min_probability','wall_speedup','solve_speedup']} for p in pairs] for case,pairs in summary.items()},indent=2))


if __name__=='__main__':
    main()
