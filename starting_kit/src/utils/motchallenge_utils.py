import re

def parse_sequence_name(file_name: str) -> str:
    # frames/yyyymmdd/clip_n_hhmm/image_0xxx.jpg
    seq = "/".join(file_name.split("/")[:-1])
    return seq.replace("/", "_")


def parse_frame_id(file_name: str) -> int:
    # image_0xxx.jpg -> xxx
    m = re.search(r'image_(\d+)\.jpg', file_name)
    if m is None:
        raise ValueError(f"Cannot parse frame id from {file_name}")
    return int(m.group(1))