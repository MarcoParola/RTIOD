"""
Robust Thermal-Image Object Detection Challenge Submission Template/Target Formatter

This is a helper script that allows you to generate your own submission or evaluation target.
The functionality can be imported or used as a standalone script.
To generate a valid submission template you need to use the official challenge json annotations.

Official annotation subsplits:
    Training  : Train.json
    Validation: Valid.json
    Testing   : Test.json (Note that Test.json only contains image data)

Usage:
    # For tempalte formatting
    python challenge_submission_formatter.py ltdv2_Valid.json

    # For evaluation target formatting
    python challenge_submission_formatter.py ltdv2_Valid.json --get_targets
"""


import json
from tqdm import tqdm
import random
from multiprocessing.pool import Pool
import pandas

__THREAD_COUNT__ = 6

def tlwh2tlbr(box):
    new_box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
    return new_box

def parse_image_uids(subset_file):
    uids = []
    # Load COCO JSON
    with open(subset_file) as f:
        data = json.load(f)
        # Parse Image IDs
        for img in data["images"]:
            uids.append(img["id"])
    return uids

def collate_dicts(dicts):
    collated_dict = {}
    for d in dicts:
        collated_dict.update(d)
        
    return collated_dict 

def generate_template(uids):
    p = Pool(__THREAD_COUNT__)
    entries = p.map(_generate_template, uids)
    template = collate_dicts(entries)
    return template

def generate_target(uids, target_json):
    uids_data = []
    # Load target COCO JSON        
    with open(target_json) as f:
        data = json.load(f)["annotations"]
        data = pandas.DataFrame(data)
        # Filter all annotations belonging to UID
        for uid in tqdm(uids, desc="Parsing annotation file"):
            uidat = {uid:{"boxes":[],"labels":[]}}
            anns = data[(data["image_id"] == uid)]
            # Parse annotations and class labels
            for idx, ann in anns.iterrows():
                uidat[uid]["boxes"].append(ann["bbox"])
                uidat[uid]["labels"].append(ann["category_id"])
            uids_data.append(uidat)
    
    #Pool and process
    p = Pool(__THREAD_COUNT__)
    entries = p.map(_generate_target, uids_data)
    template = collate_dicts(entries)
    return template

def _generate_target(uid_data):
    # Read all image UIDs and make empty template
    uid, anns = next(iter(uid_data.items()))
    content = {
        "boxes": [],
        "labels": [],
    }
    for i in range(len(anns["boxes"])):
        content["boxes"].append(tlwh2tlbr(anns["boxes"][i]))
        content["labels"].append(anns["labels"][i])
    return {uid: content}

def _generate_template(uid):
    content = {
        "boxes": [],
        "scores": [],
        "labels": [],
    }
    return {uid: content}

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser("Generate and save an empty submission json for the RWSR Challenge")
    parser.add_argument("input_json", type=str, help="Path to the dataset subset COCO file")
    parser.add_argument("--get_targets", action="store_true", help="Retrieve labels for target template")
    parser.add_argument("--output", default="./", type=str, help="Path to submission template output")
    args = parser.parse_args()

    # Load annotation file and parse UIDS
    uids = parse_image_uids(args.input_json)
    print(f"Found {len(uids)} UIDS")
    print("UID Range:")
    print(f"  {min(uids)} -> {max(uids)}")

    os.makedirs(args.output,exist_ok=True)

    # Format submission template
    if args.get_targets:
        template = generate_target(uids, args.input_json)
    else:
        template = generate_template(uids)

    # Save submission template
    if args.get_targets:
        with open(os.path.join(args.output, f"{os.path.splitext(os.path.basename(args.input_json))[0]}_target.json"), 'w') as f:
            json.dump(template, f, indent=2, sort_keys=False)
    else:
        with open(os.path.join(args.output, f"{os.path.splitext(os.path.basename(args.input_json))[0]}_template.json"), 'w') as f:
            json.dump(template, f, indent=2, sort_keys=False)
