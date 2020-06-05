import os
import argparse
import json

from torch.utils.data import DataLoader

from understanding_clouds.datasets.mask_rcnn_dataset import MaskRCNNDataset
from understanding_clouds.models.mask_rcnn.prediction import MaskRCNNPrediction
from understanding_clouds.models.mask_rcnn.model import CloudsMaskRCNN
from understanding_clouds.utils import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description='MaskRCNN for Clouds Segmentation')

    parser.add_argument('-trb', '--train_batch_size',
                        help="Trian batch size", type=int, default=1)
    parser.add_argument('-vab', '--valid_batch_size',
                        help="Trian batch size", type=int, default=1)
    parser.add_argument('--experiment_dirpath',
                        help='Where to save the model', required=True, type=str)
    parser.add_argument('--pretrained_model_path', help='Path to pretrained model',
                        type=str, default=None)
    parser.add_argument(
        '--data_path', help='Path to data', required=True, type=str)
    parser.add_argument('--subsample', default=100, type=int)
    parser.add_argument('-tts', '--train_test_split',
                        default=0.05, type=float)
    parser.add_argument('--csv_name', default='train.csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.train_test_split:
        from glob import glob
        from sklearn.model_selection import train_test_split
        df_len = len(
            glob(os.path.join(args.data_path, 'train_images/*')))
        train_ids, valid_ids = train_test_split(
            range(df_len), test_size=args.train_test_split, random_state=42)
    else:
        train_ids, valid_ids = None, None
    ds_train = MaskRCNNDataset(
        images_dirpath=args.data_path, subsample=args.subsample, split_ids=train_ids, csv_name=args.csv_name)
    ds_valid = MaskRCNNDataset(
        images_dirpath=args.data_path, subsample=args.subsample, split_ids=valid_ids, csv_name=args.csv_name)

    print('Loaded datasets!')

    dataloaders = {'TRAIN': DataLoader(ds_train, batch_size=args.train_batch_size,
                                       shuffle=False, collate_fn=collate_fn),
                   'VALID': DataLoader(ds_valid, batch_size=args.valid_batch_size,
                                       shuffle=False, collate_fn=collate_fn)}
    clouds_model = CloudsMaskRCNN(
        experiment_dirpath=args.experiment_dirpath)
    if args.pretrained_model_path is not None:
        args.pretrained_model_path = os.path.abspath(
            args.pretrained_model_path)
        clouds_model.load_model(args.pretrained_model_path)
    else:
        raise Exception('Could not load model needed for evaluation!')

    predictions = {}
    for phase, data in dataloaders.items():
        phase_preds = clouds_model.predict(data, full_pred=False)
        results = {k: v for p in phase_preds for k,
                   v in p.items()}
        predictions[phase] = results

    with open(os.path.join(args.experiment_dirpath, 'prediction_scores.json'), 'w') as f:
        json.dump(predictions, f)


if __name__ == '__main__':
    main()
