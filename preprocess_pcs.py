import os
import json
import numpy as np
from pathlib import Path


def process_pointclouds(input_dir: str, output_dir: str):
    """Normalise all .xyz point clouds and write metadata to info.json."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir_xyz = Path(output_dir/"xyz_n")
    output_dir_xyz.mkdir(parents=True, exist_ok=True)

    data_info = {}

    for xyz_file in sorted(input_dir.glob("*.xyz")):
        pts = np.loadtxt(xyz_file)[:,:3]
        #print(pts.shape)

        #center_min = pts.min(axis=0)
        #pts_shifted = pts - center_min
        #centroid = pts_shifted.mean(axis=0)
        center = (pts.min(axis=0)+pts.max(axis=0))/2. # center based on bounding box
        pts_centered = pts - center
        
        bbox_min = pts_centered.min(axis=0)
        bbox_max = pts_centered.max(axis=0)
        scale = np.linalg.norm(bbox_max - bbox_min) # should this be bbox corner ?
        pts_normalised = pts_centered / scale

        np.savetxt(output_dir_xyz / xyz_file.name, pts_normalised, fmt="%.6f")

        data_info[xyz_file.stem] = {
            "center": center.tolist(),
            "scale": float(scale),
        }

    with open(output_dir / "info.json", "w") as f:
        json.dump(data_info, f, indent=2)

    print(f"Processed {len(data_info)} files.")


#process_pointclouds("/home/jovyan/repos/review/data/Zuerich/xyz_j", "/home/jovyan/repos/review/data/Zuerich")