import json
import os
import logging
import pandas as pd
import cv2

import javabridge
import bioformats

from mescnn.detection.io.bioformats_reader import BioformatsReader
from mescnn.detection.io.openslide_reader import OpenslideReader
from mescnn.detection.qupath.paths import get_reader_type
from mescnn.detection.qupath.utils import ReaderType


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export Annotations from JSON to Image Files')
    parser.add_argument('-e', '--export', type=str, help='path/to/export', required=True)
    parser.add_argument('-w', '--wsi-dir', type=str, help='path/to/wsi/dir', required=True)
    args = parser.parse_args()

    javabridge.start_vm(class_path=bioformats.JARS)

    path_to_export = args.export
    path_to_wsi = args.wsi_dir

    path_crops = os.path.join(path_to_export, "Temp", "qu2json-output", "rois.csv")
    path_dataset = os.path.join(path_to_export, "Temp", "qu2json-output", "dataset_detectron2.json")

    df_crops = pd.read_csv(path_crops)
    with open(path_dataset, 'r') as fp:
        dataset_dict = json.load(fp)

    # dataset_dict: List[Dict<file_name, height, width, image_id, annotations>]
    # annotations: List[Dict<bbox, bbox_mode, category_id, segmentation>]

    assert len(dataset_dict) == len(df_crops), "Mismatch between crops and annotations!"

    path_to_export_json2exp = os.path.join(path_to_export, "Temp", "json2exp-output")
    os.makedirs(path_to_export_json2exp, exist_ok=True)

    path_to_original = os.path.join(path_to_export_json2exp, "Original")

    os.makedirs(path_to_original, exist_ok=True)

    for idx, row in df_crops.iterrows():
        print(f"Iter: {(idx+1):4d} / {len(df_crops):4d}")
        ext = row['ext']
        reader_type = get_reader_type(ext)
        print(f"Ext: {ext}, Reader Type: {reader_type}")

        path_to_wsi = row['path-to-wsi']
        x, y = row['x'], row['y']
        xsize, ysize = row['w'], row['h']
        idx_s = row['s']

        if reader_type in [ReaderType.SCN, ReaderType.OME_TIFF]:
            image_os = BioformatsReader(path_to_wsi)
            orig = image_os.read_resolution(image_os.indexes[idx_s], x, y, xsize, ysize, 40)
        elif reader_type in [ReaderType.NDPI, ReaderType.SVS, ReaderType.MRXS, ReaderType.TIFF]:
            image_os = OpenslideReader(path_to_wsi)
            orig = image_os.read_region((x, y), 0, (xsize, ysize))
        else:
            logging.error(f"[json2exp] reader_type '{reader_type}' invalid for wsi '{path_to_wsi}'!")
            continue

        if orig is not None:
            if orig.shape[0] != ysize or orig.shape[1] != xsize:
                logging.error(f"[json2exp] Tile [{x}, {y}, {xsize}, {ysize}] from '{path_to_wsi}' has a shape of {orig.shape}!")
                continue
        else:
            logging.error(f"[json2exp] Tile [{x}, {y}, {xsize}, {ysize}] from '{path_to_wsi}' returns None!")
            continue

        wsi_id = str(row['image-id'])
        subdir_original = os.path.join(path_to_original, wsi_id)
        os.makedirs(subdir_original, exist_ok=True)

        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        orig_file = os.path.join(subdir_original, row['filename'])
        cv2.imwrite(orig_file, orig)

    javabridge.kill_vm()
