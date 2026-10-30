#!/usr/bin/env python3
"""Four single-voxel marginals attached to the spatial SSA front map."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'paper/data/spatial_schlogl_front'
OUT=ROOT/'output/figures'
NAME='front_v64_w20_p256_marginals'
TIME=300
# Chosen from the previous spatial probability map, before viewing histograms.
VOXELS=[('A',15,'Behind the front'),('B',42,'Within the front'),
        ('C',45,'Within the front'),('D',59,'Ahead of the front')]
BIN=5


def read(suffix):
    with (DATA/(NAME+suffix)).open() as stream:return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',action='store_true',help='Render a paper-sized PDF with larger text')
    args=parser.parse_args()
    meta=json.loads((DATA/(NAME+'.json')).read_text());assert meta['status']==0
    ensemble=read('_ensemble.csv')
    selected={v:{} for _,v,_ in VOXELS}
    # Stream this potentially large output, retaining one time and four voxels.
    with (DATA/(NAME+'_marginals.csv')).open() as stream:
        for row in csv.DictReader(stream):
            if float(row['time'])==TIME and int(row['voxel'])in selected:
                selected[int(row['voxel'])][int(row['count'])]=int(row['frequency'])
    summary={};definitions='$map << EOD\n';previous=None
    for row in ensemble:
        if previous is not None and previous!=row['time']:definitions+='\n'
        definitions+=f"{float(row['voxel'])+.5} {row['time']} {row['high_probability']}\n";previous=row['time']
    definitions+='EOD\n'
    for label,v,region in VOXELS:
        counts=selected[v];assert sum(counts.values())==256
        bins=[0]*(320//BIN)
        for n,f in counts.items():
            assert n<320
            bins[n//BIN]+=f
        p_high=sum(f for n,f in counts.items()if n>=100)/256
        row=next(r for r in ensemble if float(r['time'])==TIME and int(r['voxel'])==v)
        assert abs(p_high-float(row['high_probability']))<1e-12
        mean=sum(n*f for n,f in counts.items())/256/20
        assert abs(mean-float(row['mean_concentration']))<1e-10
        definitions+=f'${label} << EOD\n'+'\n'.join(f'{BIN*(i+.5)} {f/256}'for i,f in enumerate(bins))+'\nEOD\n'
        summary[label]=dict(voxel_index=v,position=v+.5,time=TIME,region=region,
            paths=256,bin_width_molecules=BIN,high_state_probability=p_high,mean_concentration=mean,
            binned_probabilities=[f/256 for f in bins])
    # The map is placed centrally. Four bordered marginal callouts surround it.
    commands="""set encoding utf8
set multiplot
set border 3 lw 0.8
set tics out nomirror scale 0.6
set palette defined (0 'white', 1 'black')
set lmargin at screen 0.325
set rmargin at screen 0.68
set bmargin at screen 0.13
set tmargin at screen 0.88
set title 'High-state probability (256 trajectories)'
set xlabel 'Position along the chain'
set ylabel 'Time'
set xrange [0:64]
set yrange [0:320]
set xtics 16
set ytics 100
set cbrange [0:1]
set cbtics 0.5
set cblabel ''
set colorbox user origin 0.696,0.13 size 0.012,0.75
"""
    for i,(label,v,_) in enumerate(VOXELS):
        # White boxed labels locate each selected voxel on the map.
        offset=14 if label in ['A','B','D'] else -18
        commands+=f"set label {i+1} '{label}' at {v+.5},{TIME+offset} center front boxed\n"
        commands+=f"set arrow {i+1} from {v+.5},{TIME+offset*.65} to {v+.5},{TIME} nohead front lw 1 lc rgb 'black'\n"
    commands+="plot $map using 1:2:3 with image notitle\nunset label\nunset arrow\nunset colorbox\n"
    layouts=[(.055,.24,.64,.87),(.055,.24,.20,.43),(.79,.975,.64,.87),(.79,.975,.20,.43)]
    for (label,v,region),(left,right,bottom,top)in zip(VOXELS,layouts):
        y_max=max(summary[label]['binned_probabilities'])*1.18
        commands+=f"""set lmargin at screen {left}
set rmargin at screen {right}
set bmargin at screen {bottom}
set tmargin at screen {top}
set title "{label}: {region.lower()}\\nx = {v+.5:g}, t = {TIME}" offset 0,-0.4
set xlabel 'Molecules in this voxel'
set ylabel 'Probability per 5-molecule bin'
set xrange [0:280]
set yrange [0:{y_max}]
set xtics 100
set ytics autofreq
set format y '%.2f'
set boxwidth {BIN} absolute
set style fill solid 0.15 border lc rgb 'black'
plot ${label} using 1:2 with boxes lc rgb 'black' notitle
"""
    commands+="unset multiplot\n"
    destination=ROOT/'paper/figures' if args.paper else OUT
    destination.mkdir(parents=True,exist_ok=True)
    if args.paper:
        commands=commands.replace('High-state probability (256 trajectories)','High-state probability')
        commands=commands.replace('Position along the chain','Position')
        commands=commands.replace('Molecules in this voxel','Molecule count')
        commands=commands.replace('Probability per 5-molecule bin','Probability')
        commands=commands.replace('behind the front','behind front').replace('within the front','within front').replace('ahead of the front','ahead of front')
        commands=commands.replace(', t = 300','')
    terminals=[('pdf',"pdfcairo enhanced font 'Times,14' size 10in,6in")] if args.paper else [('png',"pngcairo enhanced font 'Times,17' size 2000,1200"),('pdf',"pdfcairo enhanced font 'Times,9' size 10in,6in")]
    for ext,terminal in terminals:
        target=destination/('schlogl_spatial_front_marginals.'+ext)
        subprocess.run(['gnuplot'],input=f"set terminal {terminal}\nset output '{target}'\nset style textbox opaque margins 0.2,0.1\n"+definitions+commands,text=True,check=True)
    (DATA/'marginal_insets_summary.json').write_text(json.dumps(dict(metadata=meta,selections=summary,
        interpretation='Single-voxel distributions at the same time, without pooling voxels or times. Histograms use 5-molecule bins.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
