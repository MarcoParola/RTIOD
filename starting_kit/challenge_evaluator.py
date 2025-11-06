"""
Robust Thermal-Image Object Detection Challenge Evaluator Script

This script allows you to run a local evaluation identical to the one executed on the codabench competition page.
Allowing you to validate and rank the method without using competition submissions attempts.
The scores and metrics are outputet in a "scores.json" file at in the designated output directory.
Note the competition runs the evaluation in the follwing container: https://hub.docker.com/repository/docker/asjaau/ltdv2/tags/latest/

Usage:
    python challenge_evaluator.py predictions.json valid_targets.json ./output/
"""


import os
import json
import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sys import argv

def load_template(template):
    with open(template, 'r') as f:
        dat = json.load(f)
    return dat

intervals = {
    "jan": "202101",
    "feb": "202102",
    "mar": "202103",
    "apr": "202104",
    "may": "202005",
    "jun": "202006",
    "jul": "202007",
    "aug": "202008",
}


#### INPUT/OUTPUT: Get input and output directory names
if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
    pred_file = './predictions.json'
    ref_file = './valid_targets.json'
    output_dir = './output/'
else:
    pred_file = argv[1]
    ref_file = argv[2]
    output_dir = argv[3]
    # Create the output directory, if it does not already exist and open output files
os.makedirs(output_dir, exist_ok=True)

print("########## Loading submission and target:")
submission = load_template(pred_file)
targets = load_template(ref_file)

print("########## Parsing submission:")
# Assert that templates match
if len(set(submission.keys()) ^ set(targets.keys()))!=0:
    print("Eroneous Entries:")
    u_pred = set(submission.keys()) - set(targets.keys())
    print(u_pred)
    print("Missing Entries:")
    u_targ = set(targets.keys()) - set(submission.keys())
    print(u_targ)
    raise ValueError("ERROR: Submitted template and Target template have inconsistent UIDs (See above).")
else:
    print("UUID Key pairings validated")

print("Converting to tensors")
# Convert Json elements to apropriate torch tensors
for uid in targets.keys():
    #Convert Targets
    targets[uid]["boxes"] = torch.tensor(targets[uid]["boxes"], dtype=torch.float)
    targets[uid]["labels"] = torch.tensor(targets[uid]["labels"], dtype=torch.uint8)
    #Convert Submission
    submission[uid]["boxes"] = torch.tensor(submission[uid]["boxes"], dtype=torch.float)
    submission[uid]["labels"] = torch.tensor(submission[uid]["labels"], dtype=torch.uint8)
    if "scores" in submission[uid].keys():
        submission[uid]["scores"] = torch.tensor(submission[uid]["scores"], dtype=torch.float)

print("########## Computing Metrics:")
# Initialize metrics logger
metric = MeanAveragePrecision(
    iou_type="bbox",
    box_format="xyxy",
    class_metrics=True,
    #iou_thresholds=[0.50], #Comment if COCO 0.05-0.95 is desired
    average="macro",
    ) 
scores = {}

# Compute metrics (all)
print("## 'Global' ##")
print("Processing predictions")
for entry in targets.keys():
    metric.update([submission[entry]], [targets[entry]])

print("Computing metrics")
metrics = metric.compute()
for key in metrics.keys():
    m_temp = metrics[key]
    if key in ["map","map_50","map_75", "mar_1", "mar_10", "mar_100"]:
        scores[f'global_{key}'] = float(m_temp)
    if key in ["map_per_class", "mar_10_per_class"]:
        scores[f'global_{key}'] = list(m_temp.numpy().astype(float))
    
# Compute metrics (monthly)
print("## Computing interval metrics ##")

for name, id in intervals.items():
    print(f"## '{name}' ##")
    metric.reset()
    print(f'Processing predictions: "{name}"')
    for entry in [x for x in targets.keys() if id in x[:8]]:
        metric.update([submission[entry]], [targets[entry]])

    #Calculate and store metrics
    print(f'Computing: "{name}"')
    metrics = metric.compute()
    for key in metrics.keys():
        m_temp = metrics[key]
        if key in ["map","map_50","map_75", "mar_1", "mar_10", "mar_100"]:
            scores[f'{name}_{key}'] = float(m_temp)
        if key in ["map_per_class", "mar_10_per_class"]:
            scores[f'{name}_{key}'] = list(m_temp.numpy().astype(float))

print("## Computing consistency ##")
# Compute consistency metric (Coefficient of Variation)
monthly_maps = [scores[f"{x}_map_50"] for x in intervals.keys()]
scores["global_map_con"] = float(np.std(monthly_maps, ddof=1) / np.mean(monthly_maps))
# Double check that this is the consistency metric of choice

# Compute balanced metric
print("## Computing balanced score ##")
scores["global_map_bal"] = float((1-scores["global_map_con"]) * scores["global_map_50"])

# Write scores to file
print("Saving scores to file")
print(scores)
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores, indent=4))
