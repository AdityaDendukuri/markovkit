#!/usr/bin/env python3
"""Check SSA diffusion against a binomial law and reactions against a CME."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def check(exe):
    with tempfile.TemporaryDirectory(prefix='rdme_front_') as directory:
        directory=Path(directory)
        def run(name,options):
            prefix=directory/name
            subprocess.run([str(exe),'--output-prefix',str(prefix),'--marginals','1',*options],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            ensemble=read(Path(str(prefix)+'_ensemble.csv'))
            histograms={}
            for row in read(Path(str(prefix)+'_marginals.csv')):
                histograms.setdefault((row['time'],row['voxel']),[]).append(row)
            for row in ensemble:
                hist=histograms[(row['time'],row['voxel'])]
                assert abs(sum(float(r['probability'])for r in hist)-1)<1e-12
                volume=float(options[options.index('--volume')+1])
                mean=sum(int(r['count'])*float(r['probability'])for r in hist)/volume
                high=sum(float(r['probability'])for r in hist if int(r['count'])/volume>=5)
                assert abs(mean-float(row['mean_concentration']))<1e-10
                assert abs(high-float(row['high_probability']))<1e-12
            return ensemble,read(Path(str(prefix)+'_fronts.csv'))
        rows,fronts=run('diffusion',['--cells','2','--seeded','1','--volume','1','--high','20','--low','0',
            '--paths','4000','--final-time','2','--interval','1','--jump','.5','--reaction-scale','0'])
        assert all(int(r['total_molecules'])==20 for r in fronts)
        for r in rows:
            t=float(r['time']);p=(1+math.exp(-t))/2 if r['voxel']=='0' else (1-math.exp(-t))/2
            expected=20*p
            assert abs(float(r['mean_concentration'])-expected)<6*float(r['standard_error'])+1e-10
            high=sum(math.comb(20,n)*p**n*(1-p)**(20-n) for n in range(5,21))
            assert abs(float(r['high_probability'])-high)<6*float(r['high_probability_se'])+.002
        rows,fronts=run('diffusion_2d',['--dimension','2','--cells','2','--seeded','1','--volume','1','--high','20','--low','0',
            '--paths','4000','--final-time','2','--interval','1','--jump','.5','--reaction-scale','0'])
        assert all(int(r['total_molecules'])==40 for r in fronts)
        for r in rows:
            t=float(r['time']);same=(1+math.exp(-t))/2
            px=same if int(r['voxel'])%2==0 else 1-same
            p1=px*same;p2=px*(1-same)
            expected=20*px
            assert abs(float(r['mean_concentration'])-expected)<6*float(r['standard_error'])+1e-10
            high=1-sum(math.comb(20,a)*p1**a*(1-p1)**(20-a)*math.comb(20,b)*p2**b*(1-p2)**(20-b)
                       for a in range(5)for b in range(5-a))
            assert abs(float(r['high_probability'])-high)<6*float(r['high_probability_se'])+.002
        rows,fronts=run('frozen',['--cells','2','--seeded','1','--volume','2','--high','10','--low','2',
            '--paths','2','--final-time','2','--interval','1','--jump','0','--reaction-scale','0'])
        assert all(float(r['mean_concentration'])==(10 if r['voxel']=='0' else 2) for r in rows)
        # Independent finite birth-death generator and uniformization, no diffusion.
        cap=60;birth=[.5+.17*n*(n-1)/2 for n in range(cap+1)]
        death=[.4*n+.03*n*(n-1)*(n-2)/6 for n in range(cap+1)]
        rate=max(b+d for b,d in zip(birth,death));chunks=math.ceil(rate/8);mu=rate/chunks
        p=[0.]*(cap+1);p[2]=1.
        for _ in range(chunks):
            term=p[:];weight=math.exp(-mu);result=[weight*x for x in term]
            for k in range(1,90):
                nxt=[term[n]*(1-(birth[n]+death[n])/rate) for n in range(cap+1)]
                for n in range(cap):nxt[n+1]+=birth[n]*term[n]/rate
                for n in range(1,cap+1):nxt[n-1]+=death[n]*term[n]/rate
                term=nxt;weight*=mu/k
                for n in range(cap+1):result[n]+=weight*term[n]
            p=result
        assert abs(sum(p)-1)<1e-10
        expected=sum(n*x for n,x in enumerate(p));high=sum(p[5:])
        rows,_=run('reaction',['--cells','2','--seeded','0','--volume','1','--low','2',
            '--paths','12000','--final-time','1','--interval','1','--jump','0'])
        for r in rows:
            if float(r['time'])!=1:continue
            assert abs(float(r['mean_concentration'])-expected)<6*float(r['standard_error'])+.002
            assert abs(float(r['high_probability'])-high)<6*float(r['high_probability_se'])+.002
        print(json.dumps(dict(diffusion='binomial means and high-state probabilities agree within sampling error; mass conserved',
            frozen='initial state preserved',reaction='independent uniformization CME agrees within sampling error',
            cme_mean=expected,cme_high_probability=high),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,default=ROOT/'cpp/build/schlogl_spatial_front')
    check(parser.parse_args().executable.resolve())
