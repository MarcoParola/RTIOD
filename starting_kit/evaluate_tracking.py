import os
import hydra
import pandas as pd
from src.metrics.tracking_metrics import compute_mot_metrics

@hydra.main(config_path='config', config_name='config', version_base="1.3")
def evaluate(args):

    model_name = args.modelCheckpoint.split('/')[-1].split('.')[0]
    gt_dir = os.path.join(args.tracking.trackingData, args.submission.type)
    pred_dir = os.path.join(
        args.tracking.trackingPredictionFolder, 
        args.tracking.tracker + '_' + model_name + '_th' + str(args.confidence_threshold).replace('.','_'),
        args.submission.type)
        
    per_seq_summaries = []
    seq_names = []

    for seq_file in os.listdir(gt_dir):
        if not seq_file.endswith(".txt"):
            continue

        gt_file = os.path.join(gt_dir, seq_file)
        pred_file = os.path.join(pred_dir, seq_file)

        if not os.path.exists(pred_file):
            #print(f"Missing prediction for {seq_file}")
            continue

        summary, acc = compute_mot_metrics(gt_file, pred_file)
        seq_name = os.path.splitext(seq_file)[0]
        #print(f"\nSequence: {seq_file}")
        #print(summary)

        # collect for overall aggregation
        per_seq_summaries.append(summary)
        seq_names.append(seq_name)

    # compute overall averaged metrics (macro-average: mean across sequences)
    if per_seq_summaries:
        combined = pd.concat(per_seq_summaries, axis=0)
        overall = combined.mean(axis=0)
        print("\nMacro-averaged metrics (mean across sequences):")
        print(overall)


if __name__ == "__main__":
    evaluate()
