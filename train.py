import argparse
import torch
import random
from pytorch_msssim import ssim
import torch.nn.functional as F
from torchvision.models import vgg16
from src.model import DenoisingAutoencoder
from src.data_loader import get_data_loaders
from src.constants import DEVICE, EPOCHS, N_EPOCHS_STOP, DATA_PATH, BATCH_SIZE
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Train a denoising autoencoder.')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to dataset.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs for training.')
    parser.add_argument('--n_epoch_stop', type=int, default=N_EPOCHS_STOP, help='Max epoch to wait for improvement of early stopping.')
    parser.add_argument('--model_path', type=str, default='best_dae_model.pth', help='Path to save the trained model.')
    return parser.parse_args()

wandb.init(project='denoising-autoencoder', name='acumalaka')

# Load pre-trained VGG16 model
vgg = vgg16(pretrained=True).features[:16].to(DEVICE).eval()

# Freeze VGG parameters
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(output, target):
    """Calculate perceptual loss using VGG16"""
    output_features = vgg(output)
    target_features = vgg(target)
    return F.mse_loss(output_features, target_features)

def loss_func(output, target):
    huber_loss = F.smooth_l1_loss(output, target, reduction='mean')
    l1_loss = F.l1_loss(output, target, reduction='mean')
    ssim_loss = 1 - ssim(output, target, data_range=1)
    perc_loss = perceptual_loss(output, target)
    total_loss = 0.6 * huber_loss + 0.3 * l1_loss + 0.05 * ssim_loss + 0.05 * perc_loss
    return total_loss

def train(args):
    print(f"Training model for {args.epochs} epochs")
    best_loss = float('inf')
    epochs_no_improve = 0
    dae_model = DenoisingAutoencoder(input_channels=3).to(DEVICE)
    optimizer = torch.optim.AdamW(dae_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Load data
    train_data, clean_data = get_data_loaders(data_path=args.data_path, batch_size=args.batch_size)

    # henshin to train mode
    dae_model.train()

    for epoch in range(args.epochs):
        existing_loss = 0
        for noisy, clean in zip(train_data, clean_data):
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            optimizer.zero_grad()

            denoised_img = dae_model(noisy)
            loss = loss_func(denoised_img, clean)

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=dae_model.parameters(), max_norm=1)

            optimizer.step()

        scheduler.step()

        # Early stopping
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
            # Save the model NOWW!!!
            torch.save(dae_model.state_dict(), args.model_path)
        else:
            epochs_no_improve += 1

        psnr = 10 * torch.log10(1 / loss)
        ssim_val = ssim(denoised_img, clean, data_range=1)

        wandb.log({
            'loss': loss.item(),
            'epoch': epoch,
            'psnr': psnr.item(),
            'ssim': ssim_val.item(),
            'denoised_img': wandb.Image(denoised_img[0].detach().cpu().numpy().transpose(1, 2, 0)),
        })

        print(f"epoch: {epoch}, loss: {loss:.4f}, psnr: {psnr:.4f}, ssim: {ssim_val:.4f}")

        if epochs_no_improve == args.n_epoch_stop:
            print(f'Early stopping at epoch {epoch+1} with best loss {best_loss:.4f}')
            break

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)
