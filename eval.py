import argparse
import os
import torch
import lpips
import wandb

from dataloader import ImageSet, ISTDImageSet
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import validate_model, load_checkpoint
from utils_model import get_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ifblend", help="Name of the tested model")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--decay_epoch", type=int, default=150, help="Epoch to start LR decay")
    parser.add_argument("--n_steps", type=int, default=2, help="Step decay count")

    parser.add_argument("--data_src", default=r"C:/Users/Divyam Chandak/Desktop/IFBlend/data/AMBIENT6K", help="Path to dataset")
    parser.add_argument("--res_dir", default="./final-results", help="Path to results directory")
    parser.add_argument("--ckp_dir", default="./checkpoints", help="Path to checkpoints directory")
    parser.add_argument("--load_from", default="IFBlend_ambient6k", help="Checkpoint name to load")

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()
    print(vars(opt))

    wandb.init(project="IFBLEND_evals", name=opt.load_from)
    wandb.config.update(vars(opt))

    # === Load model === #
    model_net = get_model(opt.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model_net = torch.nn.DataParallel(model_net)

    model_net = model_net.to(device)

    optimizer = torch.optim.Adam(model_net.parameters(), lr=0.0002)
    scheduler = MultiStepLR(optimizer, milestones=[], gamma=0.6)
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # === Load data === #
    if opt.data_src.endswith("6K"):
        val_dataloader = DataLoader(
            ImageSet(opt.data_src, "Test", size=None, aug=False),
            batch_size=1, shuffle=False, num_workers=8
        )
    else:
        val_dataloader = DataLoader(
            ISTDImageSet(opt.data_src, 'test', size=None, aug=False),
            batch_size=1, shuffle=False, num_workers=8
        )

    # === Sanity check === #
    if len(val_dataloader) == 0:
        print("❌ No validation data found. Please check your dataset path and structure:")
        print(f"Expected test images under: {os.path.join(opt.data_src, 'Test/input')} and 'Test/target'")
        exit(1)

    # === Load checkpoint === #
    ckp_path = os.path.join(opt.ckp_dir, opt.load_from, "best", "checkpoint.pt")
    if not os.path.exists(ckp_path):
        raise FileNotFoundError(f"❌ Checkpoint not found at {ckp_path}")

    model_net, _, _ = load_checkpoint(ckp_path, model_net, optimizer, scheduler)

    # === Validate and save results === #
    out_path = os.path.join(opt.res_dir, opt.load_from)
    os.makedirs(out_path, exist_ok=True)

    val_report = validate_model(
        model_net, val_dataloader, save_disk=True, out_dir=out_path, lpips=loss_fn
    )

    print("\n✅ Validation Results:")
    print(f"📉 MSE   : {val_report['MSE']:.4f}")
    print(f"📈 PSNR  : {val_report['PSNR']:.4f}")
    print(f"🔍 SSIM  : {val_report['SSIM']:.4f}")
    print(f"👁  LPIPS : {val_report['LPIPS']:.4f}")

    wandb.log(val_report)
    wandb.finish()
