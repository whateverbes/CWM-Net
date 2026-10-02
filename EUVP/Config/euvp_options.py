import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="../Data/EUVP/Paired/")
parser.add_argument("--checkpoints_dir", default="../ckpts/EUVP_CWM-Net/")
parser.add_argument("--pretrained", default="")
parser.add_argument("--arch", default="cwmnet", choices=["base", "wam", "cwmnet"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--learning_rate_g", type=float, default=8e-04)
parser.add_argument("--end_epoch", type=int, default=120)
parser.add_argument("--img_extension", default=".jpg")
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--wd_g", type=float, default=0.00005)
parser.add_argument("--lambda_mae", type=float, default=1.0)
parser.add_argument("--lambda_vgg", type=float, default=1.0)
parser.add_argument("--lambda_ssim", type=float, default=1.0)
parser.add_argument("--testing_dir_inp", default="../Data/EUVP/test_samples/Inp/")
parser.add_argument("--testing_dir_gt", default="../Data/EUVP/test_samples/GTr/")
parser.add_argument("--fresh", action="store_true")

opt, _ = parser.parse_known_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(opt.checkpoints_dir, exist_ok=True)
