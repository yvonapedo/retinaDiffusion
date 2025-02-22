import os
import sys
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import tempfile
from multi_task.mltdiff import GaussianDiffusionSampler, GaussianDiffusionTrainer
from multi_task.mlt_unet import UNet
from retinaDataloader import RetinaDataset
from Scheduler import GradualWarmupScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Ensure GPU is used if available


def train(modelConfig: Dict, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(os.path.join(modelConfig["dataPath"], "train.txt"), 'r') as f:
        train_areas = [line.split()[0] for line in f.readlines()]

    train_dataset = RetinaDataset(modelConfig["dataPath"], train_areas, _augment=True)

    # Use shuffle instead of DistributedSampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,  # Enable shuffling
        pin_memory=True,
        num_workers=min([os.cpu_count(), modelConfig["batch_size"], 8])
    )

    # Model setup
    model = UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    # Load pretrained weights if provided
    if modelConfig["training_load_weight"] is not None:
        model.load_state_dict(torch.load(
            os.path.join(modelConfig["save_weight_dir"], modelConfig["training_load_weight"]),
            map_location=device
        ))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4
    )
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer, multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )

    trainer = GaussianDiffusionTrainer(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]
    ).to(device)

    loss_file = open("loss.txt", "w")

    # Start training
    for e in range(modelConfig["epoch"]):
        train_one_epoch(
            model=trainer, optimizer=optimizer,
            data_loader=train_loader, device=device,
            epoch=e, modelConfig=modelConfig, f=loss_file
        )

        if (e + 1) % modelConfig["weight_save_iterval"] == 0:
            torch.save(model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], f'ckpt_{e}.pt'
            ))
        warmUpScheduler.step()

    loss_file.close()

def train_one_epoch(model, optimizer, data_loader, device, epoch, modelConfig, f):
    model.train()
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)  # Always display progress bar

    epoch_x_loss = 0.0
    epoch_img_loss = 0.0

    for step, images in enumerate(data_loader):
        x_0 = images["mask"].to(device)
        feature = images["image"].to(device)
        entropy = images["entropy"].to(device)
        x_loss, img_loss = model(x_0, feature, entropy)

        epoch_x_loss += x_loss
        epoch_img_loss += img_loss
        loss = x_loss + img_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), modelConfig["grad_clip"])
        optimizer.step()
        optimizer.zero_grad()

        # Print loss
        data_loader.desc = f"[epoch {epoch}] noise loss {epoch_x_loss.item() / (step + 1):.6f} img loss {epoch_img_loss.item() / (step + 1):.6f}"
        f.write(f"{epoch},{step},{x_loss.item()}, {img_loss.item()}\n")

    torch.cuda.synchronize(device) if device.type == "cuda" else None


if __name__ == '__main__':
    modelConfig = {
        "epoch": 100,
        "weight_save_iterval": 20,
        "batch_size": 5,
        "T": 500,
        "channel": 64,
        "channel_mult": [1, 2, 2, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "grad_clip": 1,
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints_entropy_plus_trans/",
        "dataPath": "C:/Users/yvona/Downloads/retina/Data/train",
    }

    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device id (0 or cpu)')
    args = parser.parse_args()

    train(modelConfig, args)
