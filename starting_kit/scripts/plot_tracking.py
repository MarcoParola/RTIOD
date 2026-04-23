import os
import csv
import math
import hydra
from PIL import Image, ImageDraw


def parse_pred_file(pred_file):
    # return dict: frame_idx -> list of dicts {id,x,y,w,h}
    frames = {}
    if not os.path.exists(pred_file):
        return frames
    with open(pred_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # support both comma and space separated
            if len(row) == 1:
                parts = row[0].strip().split()
            else:
                parts = row
            try:
                frame_idx = int(float(parts[0]))
                track_id = int(float(parts[1])) if len(parts) > 1 else -1
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except Exception:
                continue
            entry = {'id': track_id, 'x': x, 'y': y, 'w': w, 'h': h}
            frames.setdefault(frame_idx, []).append(entry)
    return frames


def color_for_id(track_id):
    # deterministic color map for ids
    r = (track_id * 37) % 255
    g = (track_id * 97) % 255
    b = (track_id * 59) % 255
    return (r, g, b)


def draw_overlay(img, bboxes, prev_tracks=None):
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for b in bboxes:
        tid = b['id']
        if tid is None or tid < 0:
            continue
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        # MOT-style x,y are top-left
        tl = (x, y)
        br = (x + w, y + h)
        col_rgb = color_for_id(tid)
        col = col_rgb + (180,)
        draw.rectangle([tl, br], outline=col, width=2)
        # draw id text
        txt = str(tid)
        tx, ty = tl[0] + 3, tl[1] + 3
        draw.text((tx, ty), txt, fill=(255, 255, 255, 200))
        # center
        cx = x + w / 2.0
        cy = y + h / 2.0
        draw.ellipse([(cx-3, cy-3), (cx+3, cy+3)], fill=col)

        # draw full trajectory (previous centers) if available
        if prev_tracks and tid in prev_tracks and prev_tracks[tid]:
            pts = list(prev_tracks[tid]) + [(cx, cy)]
            # draw polyline across all points
            draw.line(pts, fill=col, width=2)
            # draw previous points as small dots with lower alpha
            prev_col = col_rgb + (120,)
            for (px, py) in prev_tracks[tid]:
                draw.ellipse([(px-2, py-2), (px+2, py+2)], fill=prev_col)
    return overlay


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def plot_tracking(args):
    clip = getattr(args, 'clip', None)
    if clip is None:
        print('Please provide clip=<clip_id>')
        return

    # model name like in evaluate_tracking.py
    model_name = args.modelCheckpoint.split('/')[-1].split('.')[0]

    gt_dir = os.path.join(args.tracking.trackingData, args.submission.type)
    pred_dir = os.path.join(
        args.tracking.trackingPredictionFolder,
        args.tracking.tracker + '_' + model_name + '_th' + str(args.confidence_threshold).replace('.','_'),
        args.submission.type,
    )

    # find frames list file for this clip
    frames_dir = os.path.join(args.dataDir, 'frames')
    track_files_dir = os.path.join(args.tracking.trackingData, args.submission.type)
    matched = None
    for fn in os.listdir(track_files_dir):
        if clip in fn:
            matched = fn
            break
    if matched is None:
        print(f'No frames file found for clip {clip} in {track_files_dir}')
        return

    # parse date and clip id from filename like frames_20200514_clip_21_2239.txt
    base = os.path.splitext(matched)[0]
    parts = base.split('_')
    # parts: ['frames','20200514','clip','21','2239'] or ['frames','20200514','clip','21','2239']
    if len(parts) >= 3:
        date = parts[1]
        clip_id = '_'.join(parts[2:])
    else:
        print('Unexpected frames filename format:', matched)
        return

    clip_folder = os.path.join(frames_dir, date, clip_id)
    if not os.path.isdir(clip_folder):
        print('Clip folder not found:', clip_folder)
        return

    # list and sort frame files
    imgs = sorted([f for f in os.listdir(clip_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
    if not imgs:
        print('No images found in', clip_folder)
        return

    # predictions file matching frames file name
    pred_file = os.path.join(pred_dir, matched)
    preds = parse_pred_file(pred_file)

    out_dir = os.path.join(args.currentDir, 'outputs', 'plot', clip_id)
    os.makedirs(out_dir, exist_ok=True)

    # helpers: store full tracks as id -> list of (cx,cy)
    prev_tracks = {}
    # save index 0: first frame without bbox
    first_img = Image.open(os.path.join(clip_folder, imgs[0])).convert('RGB')
    first_img.save(os.path.join(out_dir, f"{0}.jpg"))

    # generate alpha variants from first frame with bboxes (if any)
    alphas = [i/10.0 for i in range(1, 10)]  # 0.1 .. 0.9 -> 9 images; combined with 0 gives 10
    frame_idx0 = 0
    frame_number_for_preds = frame_idx0  # assume frames start at 0
    bboxes0 = preds.get(frame_number_for_preds, [])
    for i, alpha in enumerate(alphas):
        overlay = draw_overlay(first_img, bboxes0, prev_tracks if prev_tracks else None)
        composited = Image.alpha_composite(first_img.convert('RGBA'), overlay)
        # blend for transparency
        blended = Image.blend(first_img.convert('RGBA'), composited, alpha)
        save_idx = 1 + i
        blended.convert('RGB').save(os.path.join(out_dir, f"{save_idx}.jpg"))

    next_index = 1 + len(alphas)

    # iterate through remaining frames
    for idx_in_clip, img_name in enumerate(imgs[1:], start=1):
        img_path = os.path.join(clip_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        # assume frame numbering equals idx_in_clip (starting 0)
        frame_number = idx_in_clip
        bboxes = preds.get(frame_number, [])

        # compute centers for this frame and update full tracks
        centers_this_frame = {}
        for b in bboxes:
            tid = b['id']
            if tid is None or tid < 0:
                continue
            cx = b['x'] + b['w'] / 2.0
            cy = b['y'] + b['h'] / 2.0
            centers_this_frame[tid] = (cx, cy)
            prev_tracks.setdefault(tid, [])
            # append this center to the trajectory
            prev_tracks[tid].append((cx, cy))

        overlay = draw_overlay(img, bboxes, prev_tracks)
        composited = Image.alpha_composite(img.convert('RGBA'), overlay)
        composited.convert('RGB').save(os.path.join(out_dir, f"{next_index}.jpg"))
        next_index += 1

    print('Saved plots to', out_dir)


if __name__ == '__main__':
    plot_tracking()
