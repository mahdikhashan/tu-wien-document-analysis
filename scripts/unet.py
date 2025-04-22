import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import os

PATCH_SIZE = 256
NUM_CLASSES = 1 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'unet.pth'
EVAL_OUTPUT_IMAGE_PATH = 'evaluation_output.png'


class BinarizationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augmentation=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_filename_base = os.path.splitext(self.images[index])[0]
        possible_mask_paths = [
            os.path.join(self.mask_dir, f"{mask_filename_base}.png"),
            os.path.join(self.mask_dir, f"{mask_filename_base}.bmp"),
            os.path.join(self.mask_dir, f"{mask_filename_base}.tif"),
            os.path.join(self.mask_dir, self.images[index]) # If mask has same name/ext
        ]
        mask_path = None
        for p in possible_mask_paths:
            if os.path.exists(p):
                mask_path = p
                break
        if mask_path is None:
             raise FileNotFoundError(f"Mask for {self.images[index]} not found in {self.mask_dir}")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        resize = transforms.Resize((PATCH_SIZE, PATCH_SIZE), interpolation=TF.InterpolationMode.BILINEAR) # Specify interpolation
        image = resize(image)
        mask = resize(mask)
        mask = transforms.functional.resize(mask, (PATCH_SIZE, PATCH_SIZE), interpolation=TF.InterpolationMode.NEAREST)

        if self.augmentation:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() > 0.5:
                gaussian = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                image = gaussian(image)

            if random.random() > 0.5:
                jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25) # Saturation/Hue less relevant for L
                image = jitter(image)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        mask = (mask > 0.5).float()
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # when bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits # Output raw logits

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

def combined_loss(pred, mask):
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    return bce(pred, mask) + dice(pred, mask)

def evaluate(model, image_path, mask_path, patch_size, device):
    model.eval()

    full_image = Image.open(image_path).convert("L")
    full_mask = Image.open(mask_path).convert("L")
    W, H = full_image.size

    pad_H = (patch_size - H % patch_size) % patch_size
    pad_W = (patch_size - W % patch_size) % patch_size

    padding = (0, 0, pad_W, pad_H)
    full_image_padded = TF.pad(full_image, padding, padding_mode='reflect')

    padded_W, padded_H = full_image_padded.size
    reconstructed_image = torch.zeros((1, padded_H, padded_W), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for h_idx in range(0, padded_H, patch_size):
             h_start = min(h_idx, padded_H - patch_size)
             for w_idx in range(0, padded_W, patch_size):
                w_start = min(w_idx, padded_W - patch_size)

                patch = full_image_padded.crop((w_start, h_start, w_start + patch_size, h_start + patch_size))
                patch_tensor = TF.to_tensor(patch).unsqueeze(0).to(device)

                # inference
                pred_patch_logits = model(patch_tensor)
                pred_patch_prob = torch.sigmoid(pred_patch_logits)
                
                reconstructed_image[0, h_start:h_start+patch_size, w_start:w_start+patch_size] += pred_patch_prob.squeeze(0).squeeze(0)

    reconstructed_image = reconstructed_image[:, :H, :W]
    full_mask_tensor = TF.to_tensor(full_mask) > 0.5

    pred_binary = (reconstructed_image > 0.5).float().cpu()
    target_binary = full_mask_tensor.float().cpu()

    correct = (pred_binary == target_binary).sum().item()
    accuracy = correct / (H * W)

    mse = torch.mean((pred_binary - target_binary) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        psnr = psnr.item()

    metrics = {
        "accuracy": accuracy,
        "psnr_db": psnr
    }

    print(f"Full Image Evaluation - Accuracy: {metrics['accuracy']:.4f}, PSNR: {metrics['psnr_db']:.2f} dB")
    model.train()

    return metrics, pred_binary


if __name__ == '__main__':
    train_dataset = BinarizationDataset(image_dir='data/train/img', mask_dir='data/train/gt', augmentation=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = UNet(n_channels=1, n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    criterion = combined_loss

    print("Starting Training...")
    model.train()
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 1 == 0:
                 print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"--- Epoch {epoch+1} Finished, Average Loss: {epoch_loss/len(train_loader):.4f} ---")

    print(f"\nSaving trained model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved.")

    eval_metrics, eval_output_image = evaluate(
        model,
        'input.tiff',
        'gt.tiff',
        patch_size=PATCH_SIZE,
        device=DEVICE
    )

    # save evalution result
    print(f"Saving evaluation output image to {EVAL_OUTPUT_IMAGE_PATH}")
    
    if eval_output_image.dim() == 3 and eval_output_image.shape[0] != 1 and eval_output_image.shape[0] != 3:
         eval_output_image = eval_output_image.unsqueeze(0)
    elif eval_output_image.dim() == 2: # H, W
         eval_output_image = eval_output_image.unsqueeze(0).unsqueeze(0)

    utils.save_image(eval_output_image, EVAL_OUTPUT_IMAGE_PATH)
