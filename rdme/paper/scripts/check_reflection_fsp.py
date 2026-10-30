#!/usr/bin/env python3
"""Validate exact reflection reduction against the full FSP distribution."""
import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,default=ROOT/'cpp/build/rdme_full_multigrid')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'tmp/reflection_validation')
    args=parser.parse_args()
    data=args.output_dir.resolve();data.mkdir(parents=True,exist_ok=True)
    cases={
        'schlogl':['--cells',4,'--initial',4,'--cap',10,'--steps',4],
        'two_voxels':['--cells',2,'--initial',2,'--cap',12,'--steps',4],
        '2d':['--cells',2,'--dimension',2,'--initial',4,'--cap',8,'--steps',3],
        '2d_levels':['--cells',4,'--dimension',2,'--initial',0,'--cap',2,'--linear',1],
        '3d':['--cells',2,'--dimension',3,'--initial',0,'--cap',4,'--steps',3],
        'zero_population':['--cells',4,'--initial',0,'--cap',0,'--linear',0],
        'zero_diffusion':['--cells',4,'--initial',4,'--cap',8,'--diffusion',0],
        'direct':['--cells',4,'--initial',4,'--cap',8,'--hierarchy','direct'],
        'v':['--cells',4,'--initial',4,'--cap',8,'--cycles',1],
        'k':['--cells',4,'--initial',4,'--cap',8,'--cycle-type','k'],
        'cn':['--cells',4,'--initial',4,'--cap',8,'--steps',4,'--integrator','cn'],
        'adaptive':['--cells',4,'--initial',0,'--cap',0,'--linear',1,'--steps',10,
                    '--adaptive-fsp','--fsp-tolerance','1e-5'],
        'adaptive_nonlinear':['--cells',2,'--initial',4,'--cap',4,'--shift',10,'--steps',10,
                              '--adaptive-fsp','--fsp-tolerance','1e-7'],
    }
    summary={}
    for name,options in cases.items():
        records={}
        for reflection in [0,1]:
            prefix=data/f'{name}_{reflection}'
            command=[str(args.executable.resolve()),'--cycles','2','--shift','5',
                     '--smoother','block-ilu','--sweeps','1','--tolerance','1e-12',
                     '--check','--reflection',str(reflection),*map(str,options),
                     '--state-output',str(prefix)+'_state.csv']
            with prefix.with_suffix('.csv').open('w') as out,prefix.with_suffix('.log').open('w') as err:
                status=subprocess.run(command,stdout=out,stderr=err,timeout=90).returncode
            assert status==0,(name,reflection,prefix.with_suffix('.log').read_text())
            with prefix.with_suffix('.csv').open() as stream:
                records[reflection]=list(csv.DictReader(stream))
        error=0.
        with (data/f'{name}_0_state.csv').open() as a,(data/f'{name}_1_state.csv').open() as b:
            for x,y in zip_longest(csv.DictReader(a),csv.DictReader(b)):
                assert x is not None and y is not None,name
                assert x['state']==y['state']
                error+=abs(float(x['probability'])-float(y['probability']))
        assert error<1e-10,(name,error)
        assert int(records[0][-1]['full_states'])==int(records[1][-1]['full_states'])
        if name not in ['zero_population']:
            assert int(records[1][-1]['states'])<int(records[0][-1]['states']),name
        if name.startswith('adaptive'):
            assert [(r['cap'],r['step'],r['accepted']) for r in records[0]]==[(r['cap'],r['step'],r['accepted']) for r in records[1]],name
        summary[name]=dict(joint_l1_difference=error,
            full_states=int(records[0][-1]['states']),stored_states=int(records[1][-1]['states']))
    for options,message in [(['--initial',1],'invariant initial configuration'),
                            (['--smoother','chain'],'supports point and population-block')]:
        failed=subprocess.run([str(args.executable.resolve()),'--reflection','1',*map(str,options)],
                              capture_output=True,text=True)
        assert failed.returncode!=0 and message in failed.stderr,failed.stderr
    (data/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
