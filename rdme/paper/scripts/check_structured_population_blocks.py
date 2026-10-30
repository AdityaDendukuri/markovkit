#!/usr/bin/env python3
"""Compare fused Cartesian population-block updates with generic ILU(0)."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,default=ROOT/'cpp/build/rdme_full_multigrid')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'tmp/structured_block_validation')
    args=parser.parse_args()
    data=args.output_dir.resolve()
    data.mkdir(parents=True,exist_ok=True)
    cases={
        'schlogl':['--cells',4,'--initial',4,'--cap',12,'--steps',4],
        '2d':['--cells',2,'--dimension',2,'--initial',2,'--cap',8,'--steps',3],
        '3d':['--cells',2,'--dimension',3,'--initial',2,'--cap',4,'--steps',3],
        'zero_diffusion':['--cells',4,'--initial',4,'--cap',8,'--diffusion',0],
        'linear':['--cells',16,'--initial',2,'--cap',3,'--linear',1],
        'direct':['--cells',4,'--initial',4,'--cap',8,'--hierarchy','direct'],
        'v':['--cells',4,'--initial',4,'--cap',8,'--cycles',1],
        'k':['--cells',4,'--initial',4,'--cap',8,'--cycle-type','k'],
        'gmres':['--cells',4,'--initial',4,'--cap',8,'--method','mg-gmres'],
        'cn':['--cells',4,'--initial',4,'--cap',8,'--steps',4,'--integrator','cn'],
        'adaptive':['--cells',4,'--initial',0,'--cap',0,'--linear',1,'--steps',10,
                    '--adaptive-fsp','--fsp-tolerance','1e-5'],
    }
    results={}
    for name,options in cases.items():
        outputs={}
        for kernel in ['reference','structured']:
            prefix=data/(name+'_'+kernel)
            command=[str(args.executable.resolve()),'--cycles','2','--shift','5',
                     '--smoother','block-ilu','--sweeps','1','--tolerance','1e-12',
                     '--check','--block-kernel',kernel,*map(str,options),
                     '--state-output',str(prefix)+'_state.csv']
            with prefix.with_suffix('.csv').open('w') as out,prefix.with_suffix('.log').open('w') as err:
                status=subprocess.run(command,stdout=out,stderr=err,timeout=90).returncode
            assert status==0,(name,kernel,prefix.with_suffix('.log').read_text())
            with Path(str(prefix)+'_state.csv').open() as stream:
                outputs[kernel]=[float(row['probability']) for row in csv.DictReader(stream)]
        assert len(outputs['reference'])==len(outputs['structured']),name
        difference=sum(abs(a-b) for a,b in zip(outputs['reference'],outputs['structured']))
        assert difference<1e-10,(name,difference)
        results[name]=dict(joint_l1_difference=difference,states=len(outputs['structured']))
    invalid=subprocess.run([str(args.executable.resolve()),'--block-kernel','invalid'],
                           capture_output=True,text=True)
    assert invalid.returncode!=0 and 'unknown population-block kernel' in invalid.stderr
    (data/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
