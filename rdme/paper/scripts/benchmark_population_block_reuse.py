#!/usr/bin/env python3
"""Sequential old/new timings with retained provenance and full-vector checks.

The baseline executable must be saved before rebuilding. Timings include
state output in both versions; no old published records are overwritten.
"""
import argparse
import csv
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--optimized',type=Path,default=ROOT/'cpp/build/rdme_full_multigrid')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'paper/data/population_block_reuse')
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    data=args.output_dir.resolve()
    data.mkdir(parents=True,exist_ok=True)
    cases={
        'adaptive':['--cells',4,'--initial',8,'--diffusion',1,'--integrator','be',
                    '--shift',.5,'--steps',10,'--method','mg','--cycles',2,
                    '--smoother','block-ilu','--sweeps',1,'--tolerance','1e-10',
                    '--adaptive-fsp','--cap',8,'--fsp-tolerance','1e-5'],
        'rare_fixed':['--cells',4,'--initial',0,'--diffusion',1,'--integrator','cn',
                      '--shift',.2,'--steps',50,'--method','mg','--cycles',2,
                      '--smoother','block-ilu','--sweeps',1,'--tolerance','1e-13',
                      '--cap',56,'--c1',.0258,'--c2',.0024,'--c3',.00138,'--c4',.0829],
    }
    results={}
    def run(case,version,repeat):
        exe=(args.baseline if version=='baseline' else args.optimized).resolve()
        name=f'{case}_{version}_{repeat}'
        command=[str(exe),*map(str,cases[case]),'--state-output',str(data/(name+'_state.csv'))]
        digest=hashlib.sha256(exe.read_bytes()).hexdigest()
        meta=data/(name+'.json')
        if meta.exists():
            saved=json.loads(meta.read_text())
            assert saved['command']==command and saved['executable_sha256']==digest, name
            assert saved['status']==0, name
            return saved
        print('running',name,flush=True)
        start=time.monotonic()
        status='interrupted'
        try:
            with (data/(name+'.csv')).open('w') as out,(data/(name+'.log')).open('w') as err:
                status=subprocess.run(command,stdout=out,stderr=err,timeout=1800).returncode
        finally:
            record=dict(command=command,executable_sha256=digest,status=status,
                        wall_seconds=time.monotonic()-start)
            meta.write_text(json.dumps(record,indent=2)+'\n')
        rows=list(csv.DictReader((data/(name+'.csv')).open()))
        if status==0:
            record.update(setup_seconds=float(rows[-1]['total_setup_s']) if case=='adaptive' else float(rows[0]['setup_s']),
                          peak_rss_bytes=max(int(r['peak_rss_bytes']) for r in rows),
                          cycles=sum(int(r['iterations']) for r in rows),
                          final_states=int(rows[-1]['states']),final_mass=float(rows[-1].get('mass',1-float(rows[-1].get('missing_mass',0)))))
        meta.write_text(json.dumps(record,indent=2)+'\n')
        assert status==0,(name,(data/(name+'.log')).read_text())
        print('finished',name,round(record['wall_seconds'],3),flush=True)
        return record
    for case in cases:
        results[case]=[]
        for repeat in range(args.repeats):
            pair={}
            # Alternate execution order to reduce order/temperature bias.
            for version in (['baseline','optimized'] if repeat%2==0 else ['optimized','baseline']):
                pair[version]=run(case,version,repeat)
            difference=0.
            paths=[data/f'{case}_{v}_{repeat}_state.csv' for v in ['baseline','optimized']]
            with paths[0].open() as a,paths[1].open() as b:
                for x,y in zip_longest(csv.DictReader(a),csv.DictReader(b)):
                    assert x is not None and y is not None
                    assert x['state']==y['state']
                    difference+=abs(float(x['probability'])-float(y['probability']))
            assert difference<1e-12,(case,difference)
            assert pair['baseline']['cycles']==pair['optimized']['cycles'],pair
            pair.update(joint_l1_difference=difference,
                        speedup=pair['baseline']['wall_seconds']/pair['optimized']['wall_seconds'])
            results[case].append(pair)
        (data/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
