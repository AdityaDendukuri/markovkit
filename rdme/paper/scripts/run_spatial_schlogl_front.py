#!/usr/bin/env python3
"""Reproducible exact-SSA front demonstration, separate from multigrid results."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'paper/data/spatial_schlogl_front'
EXE=ROOT/'cpp/build/schlogl_spatial_front'
DATA.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--paths',type=int,default=64)
parser.add_argument('--dimension',type=int,choices=[1,2],default=1)
parser.add_argument('--cells',type=int,default=64)
parser.add_argument('--seeded',type=int,default=16)
parser.add_argument('--final-time',type=float,default=300)
parser.add_argument('--interval',type=float,default=2)
parser.add_argument('--marginals',action='store_true')
args=parser.parse_args()
name=(f'front_v64_w20_p{args.paths}' if (args.dimension,args.cells,args.seeded,args.final_time,args.interval)==(1,64,16,300,2)
      else f'front_{args.dimension}d_c{args.cells}_seed{args.seeded}_w20_p{args.paths}_t{args.final_time:g}_obs{args.interval:g}')+('_marginals' if args.marginals else '')
meta=DATA/(name+'.json')
command=[str(EXE),'--cells',str(args.cells),'--seeded',str(args.seeded),'--volume','20','--jump','1',
         '--paths',str(args.paths),'--final-time',str(args.final_time) if args.final_time!=300 else '300','--interval',str(args.interval) if args.interval!=2 else '2','--seed','20261021',
         '--output-prefix',str(DATA/name)]
if args.dimension!=1:command+=['--dimension',str(args.dimension)]
if args.marginals:command+=['--marginals','1']
record=dict(command=command,executable_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256((ROOT/'cpp/schlogl_spatial_front.cpp').read_bytes()).hexdigest(),
            method='exact direct Gillespie SSA, not multigrid',status='running')
if meta.exists():
    saved=json.loads(meta.read_text())
    assert saved['status']==0 and saved['command']==command and saved['executable_sha256']==record['executable_sha256']
    print('using saved run')
else:
    meta.write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic()
    try:
        with (DATA/(name+'.csv')).open('w') as out,(DATA/(name+'.log')).open('w') as err:
            result=subprocess.run(command,stdout=out,stderr=err)
        record['status']=result.returncode
    except BaseException:
        record['status']='interrupted'
        raise
    finally:
        record['wall_seconds']=time.monotonic()-start
        meta.write_text(json.dumps(record,indent=2)+'\n')
    if record['status']!=0:raise RuntimeError('front simulation failed')
    print(json.dumps(record,indent=2))
