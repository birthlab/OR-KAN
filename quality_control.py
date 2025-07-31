import os
import sys
import argparse
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")

from model import OR_KAN
from dataloader import *

parser = argparse.ArgumentParser(description='MRI QC with OR-KAN, supports sequence and ori auto-detection')
parser.add_argument('--input_dir', type=str, default='./data_example_with_mask', help='Directory containing input MRI data')
parser.add_argument('--model_path', type=str, default='./checkpoint/OR_KAN_weight.pth', help='Path to the pretrained model weights')
parser.add_argument('--slice_used', type=int, default=14, help='Number of slices to use for inference')
parser.add_argument('--rotation', type=int, default=1, help='Whether to use rotation augmentation (0=No, 1=Yes)')
parser.add_argument('--sequence', type=str, choices=['TSE', 'BTFE'], default='TSE', help='MRI Sequence (TSE or BTFE)')
parser.add_argument('--ori', type=str, choices=['axial', 'coronal', 'sagittal', ''], default='', help='Orientation: axial, coronal, sagittal or auto')
args = parser.parse_args()

quality_threshold_dict = {
    ('TSE', ''): 0.42,
    ('TSE', 'axial'): 0.369,
    ('TSE', 'coronal'): 0.241,
    ('TSE', 'sagittal'): 0.537,
    ('BTFE', ''): 0.258,
    ('BTFE', 'axial'): 0.199,
    ('BTFE', 'coronal'): 0.177,
    ('BTFE', 'sagittal'): 0.510,
}

ori_index2str = {0: 'coronal', 1: 'axial', 2: 'sagittal'}
ori_str2index = {'coronal': 0, 'axial': 1, 'sagittal': 2}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def rotate_image(input_img):
    angles = torch.linspace(0, 360, 8)
    rotated_tensors = []
    rotate_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor()
    ])
    for _ in angles:
        transformed_tensor = rotate_transform(input_img.squeeze(0))
        rotated_tensors.append(transformed_tensor.unsqueeze(0))
    return torch.cat(rotated_tensors)

def calculate_predictive_entropy(predictions):
    epsilon = sys.float_info.epsilon
    predictions = torch.stack(predictions)
    mean_predictions = torch.mean(predictions, dim=0)
    predictive_entropy = -torch.sum(mean_predictions * torch.log(mean_predictions + epsilon), dim=-1)
    max_entropy = torch.log(torch.tensor(predictions.size(-1), dtype=torch.float32))
    normalized_entropy = predictive_entropy / max_entropy
    return normalized_entropy.item()

def predict_ori(output_list):
    """
    output_list: a list of length slice_used, each is a tensor shape [1,3]
    return: string, one of 'axial', 'coronal', 'sagittal'
    """
    votes = []
    for out in output_list:
        pred = out.squeeze().detach().numpy() # shape [3]
        label = np.argmax(pred)
        votes.append(label)
    vote_count = np.bincount(votes, minlength=3)
    major_ori_index = np.argmax(vote_count)
    return ori_index2str[major_ori_index]

def main():
    model = OR_KAN().to(device)
    model.load_state_dict(torch.load(args.model_path)['net'])
    model.eval()

    image_names, images = change_and_save_image_inference(args.input_dir)
    processed_images, processed_names = traverse_and_modify(images, image_names)
    test_dataset = MRIDatasetQuality_inference(processed_images, processed_names, transform=data_transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, (batch, img_path) in enumerate(test_dataloader):
        image = batch.to(device)
        output_list = []
        slice_logits_list = []
        for s in range(image.shape[1]):
            img = image[:, s:s+1, :, :].float()
            if args.rotation == 0:
                output = model(img)
                output_list.append(output.cpu())
            else:
                input_tensor = img.cpu()
                rotated_images = rotate_image(input_tensor)
                rotated_images = rotated_images.to(device)
                output = torch.mean(model(rotated_images), dim=0)
                output_list.append(output.cpu())
            slice_logits_list.append(output.cpu())

        if args.ori == '' or args.ori is None:
            pred_ori = predict_ori(slice_logits_list)
            print(f"[Auto-detected orientation for {img_path[0]}]: {pred_ori}")
            use_ori = pred_ori
        else:
            use_ori = args.ori.lower()
        
        quality_threshold = quality_threshold_dict[(args.sequence, use_ori)]

        entropy = calculate_predictive_entropy(output_list)
        quality_score = 1 - entropy
        quality_label = 'high_quality' if quality_score > quality_threshold else 'low_quality'

        print(f"{img_path[0]}: score={quality_score:.4f}, "
              f"class={quality_label}, "
              f"sequence={args.sequence}, ori={use_ori}, "
              f"threshold={quality_threshold}")

if __name__ == "__main__":
    main()
