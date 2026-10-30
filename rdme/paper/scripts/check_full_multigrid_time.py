#!/usr/bin/env python3
"""Independent small-FSP time check: CN against uniformization, in joint l1."""
import csv
import json
from run_full_multigrid import DATA,run

for mean in [8,4]:
    name=f"time_reference_m{mean}"
    run(name,["--cells",2,"--cap",40,"--shift",.01,"--method","uniformization",
        "--uniformization-mean",mean,"--state-output",DATA/f"{name}_state.csv"])
for steps in [10,20,40,80]:
    name=f"time_check_s{steps}"
    run(name,["--cells",2,"--cap",40,"--shift",2*steps/100,"--steps",steps,
        "--integrator","cn","--state-output",DATA/f"{name}_state.csv"])

def vector(name):
    return [float(r["probability"]) for r in csv.DictReader((DATA/name).open())]

a=vector("time_reference_m8_state.csv")
b=vector("time_reference_m4_state.csv")
reference_difference=sum(abs(x-y) for x,y in zip(a,b))
assert reference_difference<1e-10, reference_difference
results=[]
for steps in [10,20,40,80]:
    x=vector(f"time_check_s{steps}_state.csv")
    results.append(dict(steps=steps,dt=100/steps,joint_l1_error=sum(abs(y-z) for y,z in zip(x,b))))
result=dict(reference_difference=reference_difference,refinement=results)
(DATA/"independent_time_check.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
