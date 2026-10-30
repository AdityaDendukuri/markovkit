#!/usr/bin/env python3
"""Matched-model rare population-switching benchmark; retain all provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'paper/data/rare_switching'
RATES=[.0258,.0024,.00138,.0829]


def run(name,command):
    DATA.mkdir(parents=True,exist_ok=True)
    meta=DATA/(name+'.json')
    digest=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
    if meta.exists():
        saved=json.loads(meta.read_text())
        assert saved['command']==command and saved['executable_sha256']==digest and saved['status']==0
        print('saved',name,flush=True);return
    record=dict(command=command,executable_sha256=digest,status='running',rates=RATES,
                model='four reflecting voxels, total volume 1, D=1, initial (0,0,0,0), T=500')
    meta.write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic()
    print('running',name,flush=True)
    try:
        with (DATA/(name+'.csv')).open('w')as out,(DATA/(name+'.log')).open('w')as err:
            result=subprocess.run(command,stdout=out,stderr=err)
        record['status']=result.returncode
    except BaseException:
        record['status']='interrupted';raise
    finally:
        record['wall_seconds']=time.monotonic()-start
        meta.write_text(json.dumps(record,indent=2)+'\n')
    assert record['status']==0,record
    print('finished',name,record['wall_seconds'],flush=True)


def mg(cap,dt,hierarchy,tolerance):
    name=f'rare_peak_mg_k{cap}_dt{dt:g}_{hierarchy}_tol{tolerance:g}'
    command=[str(ROOT/'cpp/build/rdme_full_multigrid'),'--cells','4','--cap',str(cap),'--initial','0',
             '--diffusion','1','--integrator','cn','--shift',str(2/dt),'--steps',str(round(500/dt)),
             '--method','mg','--smoother','block-ilu','--sweeps','1','--cycles','2',
             '--hierarchy',hierarchy,'--tolerance',str(tolerance),'--output',str(DATA/(name+'_marginal.csv'))]
    for i,c in enumerate(RATES,1):command+=['--c'+str(i),str(c)]
    run(name,command)
    rows=list(csv.DictReader((DATA/(name+'.csv')).open()))
    assert all(r['converged']=='1'for r in rows)
    assert min(float(r['min_probability'])for r in rows)>=-1e-12
    values=[float(r['probability'])for r in csv.DictReader((DATA/(name+'_marginal.csv')).open())if float(r['time'])==500]
    print('mass',sum(values),'high N>=16',sum(values[16:]),'N20',values[20],
          'modes',[(i,values[i])for i in range(1,cap)if values[i]>values[i-1]and values[i]>values[i+1]],flush=True)


def ssa(paths,totals_only=False):
    name=f'rare_peak_ssa_p{paths}'+('_totals' if totals_only else '')
    command=[str(ROOT/'cpp/build/schlogl_spatial_front'),'--cells','4','--seeded','0',
             '--volume','.25','--jump','16','--low','0','--paths',str(paths),
             '--final-time','500','--interval','500','--seed','20261022',
             '--output-prefix',str(DATA/name)]
    for i,c in enumerate(RATES,1):command+=['--c'+str(i),str(c)]
    if totals_only:command += ['--totals-only','1']
    run(name,command)
    endpoints={};events=0
    with (DATA/(name+('_total_marginals.csv' if totals_only else '_fronts.csv'))).open()as stream:
        for r in csv.DictReader(stream):
            if float(r['time'])==500:
                n=int(r['count'] if totals_only else r['total_molecules'])
                endpoints[n]=endpoints.get(n,0)+(int(r['frequency']) if totals_only else 1)
    assert sum(endpoints.values())==paths
    summary=dict(paths=paths,high_count=sum(f for n,f in endpoints.items()if n>=16),
                 peak_bin_count=endpoints.get(20,0),endpoint_counts=endpoints)
    (DATA/(name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssa-paths',type=int,default=0)
    parser.add_argument('--totals-only',action='store_true')
    parser.add_argument('--cap',type=int,default=56)
    parser.add_argument('--dt',type=float,default=10)
    parser.add_argument('--hierarchy',choices=['recursive','direct'],default='recursive')
    parser.add_argument('--tolerance',type=float,default=1e-13)
    args=parser.parse_args()
    if args.ssa_paths:ssa(args.ssa_paths,args.totals_only)
    else:mg(args.cap,args.dt,args.hierarchy,args.tolerance)
