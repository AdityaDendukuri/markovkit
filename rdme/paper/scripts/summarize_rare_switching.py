#!/usr/bin/env python3
"""Assess rare endpoint probabilities, refinement, and plain-SSA sampling cost."""
import csv
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'paper/data/rare_switching'
OUT=ROOT/'output/figures'
NAMES=dict(coarse='rare_peak_mg_k56_dt10_recursive_tol1e-13',
           fine='rare_peak_mg_k64_dt5_recursive_tol1e-14',
           time='rare_peak_mg_k64_dt2.5_recursive_tol1e-14',
           cap='rare_peak_mg_k72_dt5_recursive_tol1e-14',
           direct='rare_peak_mg_k56_dt10_direct_tol1e-13')


def read(path):
    with path.open()as stream:return list(csv.DictReader(stream))


def mg(name):
    meta=json.loads((DATA/(name+'.json')).read_text());assert meta['status']==0
    diagnostics=read(DATA/(name+'.csv'))
    assert all(r['converged']=='1'for r in diagnostics)
    assert min(float(r['min_probability'])for r in diagnostics)>=-1e-12
    v=[float(r['probability'])for r in read(DATA/(name+'_marginal.csv'))if float(r['time'])==500]
    assert min(v)>=0
    return dict(marginal=v,mass=sum(v),missing_mass=1-sum(v),p_high=sum(v[16:]),p20=v[20],
                modes=[n for n in range(1,len(v)-1)if v[n]>v[n-1]and v[n]>v[n+1]],
                total_cycles=sum(int(r['iterations'])for r in diagnostics),states=int(diagnostics[0]['states']),
                peak_rss_bytes=max(int(r['peak_rss_bytes'])for r in diagnostics),
                max_residual_bound=max(float(r['residual_l1_bound'])for r in diagnostics),metadata=meta)


def l1(a,b):
    return sum(abs((a[i]if i<len(a)else 0)-(b[i]if i<len(b)else 0))for i in range(max(len(a),len(b))))


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [0. if k==0 else max(0,center-half),min(1,center+half)]


def main():
    results={key:mg(name)for key,name in NAMES.items()}
    ref=results['cap'];base=results['fine']
    assert 20 in ref['modes']
    comparisons={}
    for key in ['coarse','fine','time','direct']:
        r=results[key]
        comparisons[key]=dict(marginal_l1_vs_cap72=l1(r['marginal'],ref['marginal']),
                p_high_relative_difference=abs(r['p_high']/ref['p_high']-1),
                p20_relative_difference=abs(r['p20']/ref['p20']-1))
    assert comparisons['fine']['p20_relative_difference']<.01
    assert abs(results['time']['p20']/base['p20']-1)<.01
    assert ref['missing_mass']<.01*ref['p20']
    sname='rare_peak_ssa_p1000000';ssa=json.loads((DATA/(sname+'_summary.json')).read_text())
    smeta=json.loads((DATA/(sname+'.json')).read_text());assert smeta['status']==0
    M=ssa['paths'];ssa['metadata']=smeta
    for n in range(3):
        p=ref['marginal'][n];observed=ssa['endpoint_counts'].get(str(n),0)/M
        assert abs(observed-p)<6*math.sqrt(p*(1-p)/M)+1e-9, (n,observed,p)
    ssa['bulk_probability_check']='Bins N=0,1,2 agree with FSP within six predicted sampling standard errors.'
    for key,p,k in [('high',ref['p_high'],ssa['high_count']),('peak_bin',ref['p20'],ssa['peak_bin_count'])]:
        ssa[key]=dict(reference_probability=p,observed_count=k,estimate=k/M,
            expected_count=M*p,wilson_95=wilson(k,M),probability_of_zero=math.exp(M*math.log1p(-p)),
            expected_relative_standard_error=math.sqrt((1-p)/(M*p)),
            paths_for_10percent_rse=math.ceil((1-p)/(.01*p)),
            paths_for_1percent_rse=math.ceil((1-p)/(.0001*p)))
    matched_meta=DATA/'rare_peak_ssa_p60000000_totals.json'
    matched_summary=DATA/'rare_peak_ssa_p60000000_totals_summary.json'
    matched=None
    if matched_summary.exists() and json.loads(matched_meta.read_text())['status']==0:
        matched=json.loads(matched_summary.read_text())
        matched['metadata']=json.loads(matched_meta.read_text())
        for key,p,k in [('high',ref['p_high'],matched['high_count']),('peak_bin',ref['p20'],matched['peak_bin_count'])]:
            size=matched['paths']
            matched[key]=dict(reference_probability=p,observed_count=k,estimate=k/size,
                 expected_relative_standard_error=math.sqrt((1-p)/(size*p)),
                 observed_relative_standard_error=math.sqrt((1-k/size)/k) if k else None,
                 wilson_95=wilson(k,size))
    summary=dict(results=results,refinement=comparisons,ssa=ssa,matched_ssa=matched,
        distinction='Endpoint occupancy at T=500, not first passage. Refinement differences are diagnostics, not a certified total transient error bound.')
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    OUT.mkdir(parents=True,exist_ok=True)
    display_M=matched['paths'] if matched else M
    display_counts={int(n):int(k)for n,k in (matched if matched else ssa)['endpoint_counts'].items()}
    additional_points=", $comparison using 1:($2>0?$2:1/0) with points lc rgb 'black' pt 4 ps 0.65 title '60,000,000 SSA paths'" if matched else ''
    definitions='$fsp << EOD\n'+'\n'.join(f'{n} {p:.17g} {display_M*p:.17g}'for n,p in enumerate(ref['marginal']))+'\nEOD\n'
    counts={int(n):int(k)for n,k in ssa['endpoint_counts'].items()}
    definitions+='$ssa << EOD\n'+'\n'.join(f'{n} {counts.get(n,0)/M:.17g} {counts.get(n,0)}'for n in range(73))+'\nEOD\n'
    definitions+='$comparison << EOD\n'+'\n'.join(f'{n} {display_counts.get(n,0)/display_M:.17g} {display_counts.get(n,0)}'for n in range(73))+'\nEOD\n'
    ymax=max([display_M*p for p in ref['marginal'][12:33]]+[display_counts.get(n,0)for n in range(12,33)])*1.25
    commands=f"""set encoding utf8
set border 3 lw 0.8
set tics nomirror out scale 0.6
set style line 1 lc rgb 'black' lw 1.8 dt 1
set style line 2 lc rgb 'black' pt 6 ps 0.7
set key top right
set multiplot
set lmargin at screen 0.095
set rmargin at screen 0.47
set bmargin at screen 0.16
set tmargin at screen 0.87
set title '(a) Total-population distribution at t = 500'
set xlabel 'Total molecule count, k'
set ylabel 'Probability'
set logscale y
set format y '10^{{%L}}'
set yrange [1e-10:2]
set xrange [0:55]
plot $fsp using 1:2 with lines ls 1 title 'Multigrid FSP', $ssa using 1:($2>0?$2:1/0) with points ls 2 title '1,000,000 SSA paths'{additional_points}
unset logscale y
set format y '%g'
set lmargin at screen 0.66
set rmargin at screen 0.98
set bmargin at screen 0.63
set tmargin at screen 0.87
set title '(b) Small secondary maximum (FSP)'
set xlabel 'Total molecule count, k'
set ylabel 'Probability (x 10^{{-7}})'
set xrange [12:24]
set yrange [1.05:1.3]
set xtics 4
set ytics 0.1
unset key
plot $fsp using 1:($2*1e7) with lines ls 1 notitle
set lmargin at screen 0.66
set rmargin at screen 0.98
set bmargin at screen 0.16
set tmargin at screen 0.40
set ytics autofreq
set key top right
set title '(c) Rare bins in {display_M:,} SSA paths'
set xlabel 'Total molecule count, k'
set ylabel 'Endpoint count'
set xrange [12:32]
set yrange [0:{ymax}]
set xtics 4
set boxwidth 0.8 absolute
set style fill solid 0.15 border lc rgb 'black'
plot $comparison using 1:3 with boxes lc rgb 'black' title 'Observed SSA counts', $fsp using 1:3 with lines ls 1 title 'Expected counts from FSP'
unset multiplot
"""
    for dest in [OUT,ROOT/'paper/figures']:
        for ext,terminal in [('pdf',"pdfcairo enhanced font 'Times,11' size 7.5in,5in")]+([]if dest!=OUT else [('png',"pngcairo enhanced font 'Times,19' size 1900,1267")]):
            target=dest/('rdme_rare_switching.'+ext)
            subprocess.run(['gnuplot'],input=f"set terminal {terminal}\nset output '{target}'\n"+definitions+commands,text=True,check=True)
    report=['# Rare endpoint probabilities in the four-voxel RDME','','See BENCHMARK_DESIGN.md for the complete model and screening record. Final rate constants are (0.0258, 0.0024, 0.00138, 0.0829); D=1; initial (0,0,0,0); T=500. All results below solve or simulate the identical spatial model.','',
        '## Multigrid and numerical refinement','','| Run | States | Wall time (s) | P(N >= 16) | P(N = 20) | Missing mass |',
        '|---|---:|---:|---:|---:|---:|']
    for key,r in results.items():report.append(f"| {key} | {r['states']} | {r['metadata']['wall_seconds']:.3f} | {r['p_high']:.12g} | {r['p20']:.12g} | {r['missing_mass']:.3g} |")
    report+=['','Reference for the comparisons is cap 72, dt=5. The time refinement uses cap 64, dt=2.5. Coarse and direct use cap 56, dt=10; fine uses cap 64, dt=5. Both recursive and direct use the same population-block relaxation, W coarse visits, shift, tolerance, and initial iterate. Timings are sequential single runs including output.','',
        'These are numerical cap/time refinement diagnostics. The per-solve residual bound and reported missing mass are not a rigorous bound on the complete continuous-time error. No state clipping or renormalization is used.','',
        '## Completed SSA run','',f"One million independent trajectories completed in {smeta['wall_seconds']:.3f} s.", '',
        '| Observable | FSP probability | Expected samples | Observed samples | Expected SSA relative standard error | Paths for 10% relative standard error |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ['high','peak_bin']:
        r=ssa[key];report.append(f"| {key} | {r['reference_probability']:.9g} | {r['expected_count']:.3f} | {r['observed_count']} | {r['expected_relative_standard_error']:.3f} | {r['paths_for_10percent_rse']:,} |")
    report+=['','Sample requirements are computed from the Bernoulli variance using the refined FSP probabilities. They are predictions, not completed Monte Carlo counts or guaranteed confidence levels. Exact SSA samples paths without time-discretization bias; its endpoint probabilities have sampling error. The comparison concerns ordinary independent SSA, not specialized rare-event sampling.','',
        'The central numerical result is the resolved positive small secondary maximum at N=20. The low-population mode is N=0. The high-branch maximum is shallow, and the figure should not exaggerate its prominence. No stationary, first-passage, or general SSA-superiority claim follows from this experiment.']
    if matched:
        report+=['','## Completed SSA run at the target basin accuracy','',
            f"{matched['paths']:,} independent SSA trajectories completed in {matched['metadata']['wall_seconds']:.3f} s, retaining aggregate endpoint counts. This output mode was checked to reproduce the first million raw trajectory endpoints exactly. The larger ensemble contains the first million paths; the two ensembles are not independent additional samples.",'',
            f"High-region endpoints: {matched['high_count']}; estimated probability {matched['high']['estimate']:.12g}; expected relative standard error {matched['high']['expected_relative_standard_error']:.4f}.",'',
            f"Endpoints in the N=20 bin: {matched['peak_bin_count']}; estimated probability {matched['peak_bin']['estimate']:.12g}; expected relative standard error {matched['peak_bin']['expected_relative_standard_error']:.4f}.",'',
            'The sample size was set from the previously refined FSP high-region probability before viewing these endpoint counts. The expected high-region relative standard error is below 10%. The individual rare-bin probability remains much less precisely estimated.']
    (DATA/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(refinement=comparisons,ssa={k:ssa[k]for k in ['high','peak_bin']},timings={k:r['metadata']['wall_seconds']for k,r in results.items()}),indent=2))


if __name__=='__main__':main()
