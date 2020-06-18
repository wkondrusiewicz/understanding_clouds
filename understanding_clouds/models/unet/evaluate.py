import os
import argparse
import json

from torch.utils.data import DataLoader

from understanding_clouds.datasets.unet_dataset import UnetDataset
from understanding_clouds.models.unet.unet import CloudsUnet
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
    ds_train = UnetDataset(images_dirpath=args.data_path, subsample=args.subsample, split_ids=train_ids)
    ds_valid = UnetDataset(images_dirpath=args.data_path, subsample=args.subsample, split_ids=valid_ids)

    print('Loaded datasets!')

    dataloaders = {'TRAIN': DataLoader(ds_train, batch_size=args.train_batch_size,
                                       shuffle=False),
                   'VALID': DataLoader(ds_valid, batch_size=args.valid_batch_size,
                                       shuffle=False)}
    clouds_model = CloudsUnet(
        experiment_dirpath=args.experiment_dirpath)
    if args.pretrained_model_path is not None:
        args.pretrained_model_path = os.path.abspath(
            args.pretrained_model_path)
        clouds_model.load_model(args.pretrained_model_path)
    else:
        raise Exception('Could not load model needed for evaluation!')

    predictions = {}
    for phase, data in dataloaders.items():
        phase_preds = clouds_model.predict(data, scores_only=True)
        results = {f: s for f, s in phase_preds}
        predictions[phase] = results

    with open(os.path.join(args.experiment_dirpath, 'prediction_scores.json'), 'w') as f:
        json.dump(predictions, f)


if __name__ == '__main__':
    main()
