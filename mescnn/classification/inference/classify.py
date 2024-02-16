import os
import numpy as np
import torch
import pandas as pd

from mmcls.apis import init_model
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

from mescnn.classification.gutils.utils import get_proper_device
from mescnn.classification.inference.paths import get_logs_path
from mescnn.definitions import ROOT_DIR


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Classifiers Inference for Glomeruli Task')
    parser.add_argument('-r', '--root-path', type=str, help='Root path', default=ROOT_DIR)
    parser.add_argument('-e', '--export-dir', type=str, help='Directory to export report', required=True)
    parser.add_argument('--netB', type=str, help='Network architecture for Sclerotic vs. Non-Sclerotic', required=True)
    parser.add_argument('--netM', type=str, help='Network architecture for 12 classes (Non-sclerotic)', required=True)
    args = parser.parse_args()

    net_name_dict = {
        "B": args.netB,
        "M": args.netM,
    }

    criterion_pr = "min"
    root_path = args.root_path
    export_dir = args.export_dir

    mesc_log_dir = get_logs_path(root_path)
    crop_dir = os.path.join(export_dir, "Temp", "json2exp-output", "Original")
    # report_dir = os.path.join(export_dir, "Report")
    report_dir = os.path.join(export_dir, "Report", f"B-{args.netB}_M-{args.netM}")
    os.makedirs(report_dir, exist_ok=True)

    wsi_ids = os.listdir(crop_dir)
    logs_path = get_logs_path(ROOT_DIR)
    subdir = "test-res"

    wsi_dict = {
        'WSI-ID': [],
        'most-predicted-class': [],
        'ratio-most-predicted-class': [],
    }

    output_file_summary_csv = os.path.join(report_dir, "summary.csv")

    for wsi_id in wsi_ids:
        output_file_csv = os.path.join(report_dir, f"{wsi_id}.csv")
        prediction_dir = os.path.join(crop_dir, wsi_id)

        mesc_dict = {
            'filename': [],
            'predicted-class': [],

            'NoSclerotic-prob': [],
            'Sclerotic-prob': [],

            'ABMGN-prob': [],
            'ANCA-prob': [],
            'C3-GN-prob': [],
            'CryoglobulinemicGN-prob': [],
            'DDD-prob': [],
            'Fibrillary-prob': [],
            'IAGN-prob': [],
            'IgAGN-prob': [],
            'MPGN-prob': [],
            'Membranous-prob': [],
            'PGNMID-prob': [],
            'SLEGN-IV-prob': [],
        }

        # Model 1: Sclerotic vs. Non-Sclerotic
        net_name = net_name_dict["B"]
        net_path = os.path.join(mesc_log_dir, 'binary', net_name, f'{net_name}_B_ckpt.pth')
        config_path = os.path.join(mesc_log_dir, 'binary', net_name, f'{net_name}_B_config.py')

        device = get_proper_device()
        bin_model = init_model(config_path, net_path, device=device)

        # Model 2: 12 classes
        net_name = net_name_dict["M"]
        net_path = os.path.join(mesc_log_dir, '12classes', net_name, f'{net_name}_M_ckpt.pth')
        config_path = os.path.join(mesc_log_dir, '12classes', net_name, f'{net_name}_M_config.py')
        
        mult_model = init_model(config_path, net_path, device=device)
        
        images_list = os.listdir(prediction_dir)
        images_list = [os.path.join(prediction_dir, f) for f in images_list if f.endswith(".jpeg")]
        for image_path in images_list:
            # Build the data pipeline
            data = dict(img_info=dict(filename=image_path), img_prefix=None)
            pipeline = bin_model.cfg.data.test.pipeline
            comp_pipeline = Compose(pipeline)
            data = comp_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            if next(bin_model.parameters()).is_cuda:
                # Scatter to specified GPU
                data = scatter(data, [device])[0]

            # Forward the sclerotic vs. non-sclerotic model
            with torch.no_grad():
                scores = bin_model(return_loss=False, **data)
            # Collect the predicted class and the scores
            pred_label = np.argsort(scores, axis=1)[0][::-1]
            pred_class = bin_model.CLASSES[pred_label[0]][3:]
            scores = scores[0]
            mesc_dict['NoSclerotic-prob'].append(scores[0])
            mesc_dict['Sclerotic-prob'].append(scores[1])

            if pred_class == "Sclerotic":
                # Append NaNs for the 12 classes
                mesc_dict['ABMGN-prob'].append(np.nan)
                mesc_dict['ANCA-prob'].append(np.nan)
                mesc_dict['C3-GN-prob'].append(np.nan)
                mesc_dict['CryoglobulinemicGN-prob'].append(np.nan)
                mesc_dict['DDD-prob'].append(np.nan)
                mesc_dict['Fibrillary-prob'].append(np.nan)
                mesc_dict['IAGN-prob'].append(np.nan)
                mesc_dict['IgAGN-prob'].append(np.nan)
                mesc_dict['MPGN-prob'].append(np.nan)
                mesc_dict['Membranous-prob'].append(np.nan)
                mesc_dict['PGNMID-prob'].append(np.nan)
                mesc_dict['SLEGN-IV-prob'].append(np.nan)
            else:
                # Forward the 12 classes model
                with torch.no_grad():
                    scores = mult_model(return_loss=False, **data)
                # Collect the predicted class and the scores
                pred_label = np.argsort(scores, axis=1)[0][::-1]
                pred_class = mult_model.CLASSES[pred_label[0]][3:]
                scores = scores[0]
                mesc_dict['ABMGN-prob'].append(scores[0])
                mesc_dict['ANCA-prob'].append(scores[1])
                mesc_dict['C3-GN-prob'].append(scores[2])
                mesc_dict['CryoglobulinemicGN-prob'].append(scores[3])
                mesc_dict['DDD-prob'].append(scores[4])
                mesc_dict['Fibrillary-prob'].append(scores[5])
                mesc_dict['IAGN-prob'].append(scores[6])
                mesc_dict['IgAGN-prob'].append(scores[7])
                mesc_dict['MPGN-prob'].append(scores[8])
                mesc_dict['Membranous-prob'].append(scores[9])
                mesc_dict['PGNMID-prob'].append(scores[10])
                mesc_dict['SLEGN-IV-prob'].append(scores[11])
            
            mesc_dict['filename'].append(image_path)
            mesc_dict['predicted-class'].append(pred_class)

        mesc_df = pd.DataFrame(data=mesc_dict)
        mesc_df.to_csv(output_file_csv, sep=';', index=False)

        # Save each WSI's results
        most_predicted_class = mesc_df['predicted-class'].mode().values[0]
        count_most_predicted_class = mesc_df[mesc_df['predicted-class'] == most_predicted_class].shape[0]
        total_crops = mesc_df.shape[0]

        wsi_dict['WSI-ID'].append(wsi_id)
        wsi_dict['most-predicted-class'].append(most_predicted_class)
        wsi_dict['ratio-most-predicted-class'].append(f'{count_most_predicted_class} | {total_crops}')


    wsi_df = pd.DataFrame(data=wsi_dict)
    wsi_df.to_csv(output_file_summary_csv, sep=';', index=False)
