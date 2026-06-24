"""
run.py — A4 Generative Models Training Script
==============================================
Usage examples:
    python3 run.py --model gan      --dataset mnist  --epochs 20 --train
    python3 run.py --model cyclegan --dataset celeba --epochs 20 --train
    python3 run.py --model ddpm     --dataset mnist  --epochs 20 --train
    python3 run.py --model ddpm     --dataset mnist  --epochs 20 --schedule cosine --train
    python3 run.py --model cyclegan --weights saved/cyclegan_celeba.pt --test-image my_face.jpg
    python3 run.py --model ddpm     --weights saved/ddpm_mnist.pt --generate --n 64
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()
os.makedirs('saved', exist_ok=True)
os.makedirs('figs',  exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

def savefig(name):
    path = f'figs/{name}.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved figure → {path}")

def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── GAN ───────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),    nn.LeakyReLU(0.2),
            nn.Linear(256, 512),      nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),     nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim), nn.Tanh(),
        )
    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512),     nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),      nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),        nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


# ── CycleGAN ──────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )
    def forward(self, x): return x + self.block(x)


class CycleGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_res=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),   nn.InstanceNorm2d(ngf),   nn.ReLU(True),
            nn.Conv2d(ngf,   ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),   nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),   nn.ReLU(True),
        ]
        for _ in range(n_res):
            layers.append(ResidualBlock(ngf * 4))
        layers += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),   nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7), nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        def block(ic, oc, norm=True):
            l = [nn.Conv2d(ic, oc, 4, stride=2, padding=1)]
            if norm: l.append(nn.InstanceNorm2d(oc))
            l.append(nn.LeakyReLU(0.2, inplace=True))
            return l
        self.model = nn.Sequential(
            *block(in_ch, ndf, norm=False),
            *block(ndf,   ndf*2),
            *block(ndf*2, ndf*4),
            nn.ZeroPad2d(1),
            nn.Conv2d(ndf*4, 1, 4, padding=1),
        )
    def forward(self, x): return self.model(x)


# ── DDPM ──────────────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device).float()
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1    = nn.GroupNorm(8, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)
    def forward(self, x, t_emb):
        h = self.norm1(self.conv1(x) * torch.sigmoid(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.norm2(self.conv2(h) * torch.sigmoid(self.conv2(h)))
        return h + self.residual(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc1 = ResBlock(in_ch,      base_ch,   time_dim)
        self.enc2 = ResBlock(base_ch,    base_ch*2, time_dim)
        self.down = nn.MaxPool2d(2)
        self.bot  = ResBlock(base_ch*2,  base_ch*4, time_dim)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ResBlock(base_ch*4 + base_ch*2, base_ch*2, time_dim)
        self.dec1 = ResBlock(base_ch*2 + base_ch,   base_ch,   time_dim)
        self.out  = nn.Conv2d(base_ch, in_ch, 1)
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        e1 = self.enc1(x,             t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        b  = self.bot( self.down(e2), t_emb)
        d2 = self.dec2(torch.cat([self.up(b),  e2], 1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1), t_emb)
        return self.out(d1)


# ══════════════════════════════════════════════════════════════════════════════
#  NOISE SCHEDULES
# ══════════════════════════════════════════════════════════════════════════════
T = 1000

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """Nichol & Dhariwal (2021) cosine schedule."""
    t = torch.linspace(0, timesteps, timesteps + 1)
    alphas_bar = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def build_schedule(schedule='linear'):
    fn    = cosine_beta_schedule if schedule == 'cosine' else linear_beta_schedule
    betas = fn(T).to(device)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab   = torch.sqrt(alpha_bar)
    sqrt_1mab = torch.sqrt(1.0 - alpha_bar)
    return betas, alphas, alpha_bar, sqrt_ab, sqrt_1mab

def q_sample(x0, t, sqrt_ab, sqrt_1mab, noise=None):
    if noise is None: noise = torch.randn_like(x0)
    return sqrt_ab[t][:,None,None,None] * x0 + sqrt_1mab[t][:,None,None,None] * noise


# ══════════════════════════════════════════════════════════════════════════════
#  CELEBA CUSTOM DATASET  (Kaggle CSV format)
# ══════════════════════════════════════════════════════════════════════════════
class CelebAHairDataset(Dataset):
    def __init__(self, img_dir, attr_df, indices, transform=None):
        self.img_dir   = img_dir
        self.attr_df   = attr_df
        self.indices   = indices
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        row  = self.attr_df.iloc[self.indices[idx]]
        img  = Image.open(os.path.join(self.img_dir, row['image_id'])).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, 0

def load_celeba(data_root='./data', max_per_class=5000, batch_size=16, num_workers=4):
    import pandas as pd
    IMG_SIZE = 64
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    attr_df  = pd.read_csv(os.path.join(data_root, 'celeba', 'list_attr_celeba.txt'))
    img_dir  = os.path.join(data_root, 'celeba', 'img_align_celeba')
    blonde_idx = attr_df.index[attr_df['Blond_Hair'] ==  1].tolist()[:max_per_class]
    dark_idx   = attr_df.index[attr_df['Blond_Hair'] == -1].tolist()[:max_per_class]
    dark_ds   = CelebAHairDataset(img_dir, attr_df, dark_idx,   transform=transform)
    blonde_ds = CelebAHairDataset(img_dir, attr_df, blonde_idx, transform=transform)
    ld = DataLoader(dark_ds,   batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    lb = DataLoader(blonde_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    print(f"Dark hair   : {len(dark_ds)} images")
    print(f"Blonde hair : {len(blonde_ds)} images")
    return dark_ds, blonde_ds, ld, lb


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_gan(epochs=20, data_root='./data', num_workers=4):
    print(f"\n{'='*60}")
    print(f"  Training Vanilla GAN on MNIST for {epochs} epochs")
    print(f"{'='*60}")
    set_seed()
    Z_DIM = 100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])
    loader = DataLoader(
        torchvision.datasets.MNIST(data_root, train=True, download=True, transform=transform),
        batch_size=128, shuffle=True, num_workers=num_workers,
    )
    G = Generator(Z_DIM).to(device)
    D = Discriminator().to(device)
    opt_G     = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D     = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_z   = torch.randn(64, Z_DIM, device=device)

    g_losses, d_losses = [], []
    for epoch in range(epochs):
        t0 = time.time(); g_ep = []; d_ep = []
        G.train(); D.train()
        for real, _ in tqdm(loader, desc=f'GAN Epoch {epoch+1}/{epochs}', leave=False):
            B    = real.size(0)
            real = real.view(B, -1).to(device)
            ones  = torch.ones(B,  1, device=device)
            zeros = torch.zeros(B, 1, device=device)
            z = torch.randn(B, Z_DIM, device=device)
            d_loss = criterion(D(real), ones) + criterion(D(G(z).detach()), zeros)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            z = torch.randn(B, Z_DIM, device=device)
            g_loss = criterion(D(G(z)), ones)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            g_ep.append(g_loss.item()); d_ep.append(d_loss.item())

        g_losses.append(np.mean(g_ep)); d_losses.append(np.mean(d_ep))
        print(f"Epoch {epoch+1:02d} | G:{np.mean(g_ep):.3f} D:{np.mean(d_ep):.3f} | {time.time()-t0:.1f}s")

        if (epoch + 1) % 5 == 0:
            G.eval()
            with torch.no_grad():
                fake = G(fixed_z).view(-1,1,28,28).cpu()
            grid = torchvision.utils.make_grid(fake, nrow=8, normalize=True)
            plt.figure(figsize=(8,8))
            plt.imshow(grid.permute(1,2,0)); plt.axis('off')
            plt.title(f'GAN Epoch {epoch+1}')
            savefig(f'gan_epoch{epoch+1}')
            G.train()

    torch.save(G.state_dict(), 'saved/gan_mnist.pt')
    print("Saved → saved/gan_mnist.pt")

    # Loss plot
    plt.figure(figsize=(8,4))
    plt.plot(g_losses, label='Generator',     color='steelblue')
    plt.plot(d_losses, label='Discriminator', color='coral')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Vanilla GAN Training Losses')
    plt.legend(); plt.grid(alpha=0.4)
    savefig('gan_loss_curves')


def train_cyclegan(epochs=20, data_root='./data', num_workers=4,
                   lambda_cyc=10.0, lambda_idt=5.0):
    print(f"\n{'='*60}")
    print(f"  Training CycleGAN on CelebA for {epochs} epochs")
    print(f"  λ_cyc={lambda_cyc}  λ_idt={lambda_idt}")
    print(f"{'='*60}")
    set_seed()
    _, _, loader_dark, loader_blonde = load_celeba(
        data_root=data_root, num_workers=num_workers
    )
    G   = CycleGenerator().to(device)
    Fg  = CycleGenerator().to(device)
    D_X = PatchDiscriminator().to(device)
    D_Y = PatchDiscriminator().to(device)
    opt_G_all = torch.optim.Adam(
        list(G.parameters()) + list(Fg.parameters()), lr=2e-4, betas=(0.5,0.999))
    opt_D_all = torch.optim.Adam(
        list(D_X.parameters()) + list(D_Y.parameters()), lr=2e-4, betas=(0.5,0.999))
    adv_fn = nn.MSELoss()
    cyc_fn = nn.L1Loss()

    g_losses, d_losses = [], []
    for epoch in range(epochs):
        t0 = time.time(); g_ep = []; d_ep = []
        dark_it   = iter(loader_dark)
        blonde_it = iter(loader_blonde)
        n_batches = min(len(loader_dark), len(loader_blonde))

        for _ in tqdm(range(n_batches), desc=f'CycleGAN Epoch {epoch+1}/{epochs}', mininterval=5.0):
            rx, _ = next(dark_it); ry, _ = next(blonde_it)
            rx, ry = rx.to(device), ry.to(device)

            opt_G_all.zero_grad()
            fake_y = G(rx);  fake_x = Fg(ry)
            cy_x   = Fg(fake_y); cy_y = G(fake_x)
            idt_x  = Fg(rx);    idt_y = G(ry)

            pshape   = D_Y(fake_y).shape
            real_lbl = torch.ones(pshape,  device=device)
            fake_lbl = torch.zeros(pshape, device=device)

            l_adv = adv_fn(D_Y(fake_y), real_lbl) + adv_fn(D_X(fake_x), real_lbl)
            l_cyc = cyc_fn(cy_x, rx) + cyc_fn(cy_y, ry)
            l_idt = cyc_fn(idt_x, rx) + cyc_fn(idt_y, ry)
            l_G   = l_adv + lambda_cyc*l_cyc + lambda_idt*l_idt
            l_G.backward(); opt_G_all.step()

            opt_D_all.zero_grad()
            l_DX = adv_fn(D_X(rx), real_lbl) + adv_fn(D_X(fake_x.detach()), fake_lbl)
            l_DY = adv_fn(D_Y(ry), real_lbl) + adv_fn(D_Y(fake_y.detach()), fake_lbl)
            l_D  = (l_DX + l_DY) * 0.5
            l_D.backward(); opt_D_all.step()

            g_ep.append(l_G.item()); d_ep.append(l_D.item())

        g_losses.append(np.mean(g_ep)); d_losses.append(np.mean(d_ep))
        print(f"Epoch {epoch+1:02d} | G:{np.mean(g_ep):.3f} D:{np.mean(d_ep):.3f} | {time.time()-t0:.1f}s")

    torch.save({'G': G.state_dict(), 'F': Fg.state_dict()}, 'saved/cyclegan_celeba.pt')
    print("Saved → saved/cyclegan_celeba.pt")

    plt.figure(figsize=(8,4))
    plt.plot(g_losses, label='Generator',     color='steelblue')
    plt.plot(d_losses, label='Discriminator', color='coral')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('CycleGAN Training Losses')
    plt.legend(); plt.grid(alpha=0.4)
    savefig('cyclegan_loss_curves')


def train_ddpm(epochs=20, data_root='./data', num_workers=4, schedule='linear'):
    print(f"\n{'='*60}")
    print(f"  Training DDPM on MNIST for {epochs} epochs  [{schedule} schedule]")
    print(f"{'='*60}")
    set_seed()
    betas, alphas, alpha_bar, sqrt_ab, sqrt_1mab = build_schedule(schedule)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])
    loader = DataLoader(
        torchvision.datasets.MNIST(data_root, train=True, download=True, transform=transform),
        batch_size=128, shuffle=True, num_workers=num_workers,
    )
    unet = SimpleUNet().to(device)
    opt  = torch.optim.Adam(unet.parameters(), lr=2e-4)
    print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")

    losses = []
    for epoch in range(epochs):
        unet.train(); ep_loss = []
        for x0, _ in tqdm(loader, desc=f'DDPM [{schedule}] Epoch {epoch+1}/{epochs}', leave=False):
            x0    = x0.to(device)
            B     = x0.size(0)
            t     = torch.randint(0, T, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t   = q_sample(x0, t, sqrt_ab, sqrt_1mab, noise)
            pred  = unet(x_t, t)
            loss  = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss.append(loss.item())
        losses.append(np.mean(ep_loss))
        print(f'Epoch {epoch+1:02d} | Loss: {np.mean(ep_loss):.4f}')

    save_path = f'saved/ddpm_mnist.pt' if schedule == 'linear' else f'saved/ddpm_{schedule}.pt'
    torch.save(unet.state_dict(), save_path)
    print(f"Saved → {save_path}")

    plt.figure(figsize=(8,4))
    plt.plot(losses, marker='o', color='orange')
    plt.title(f'DDPM Training Loss [{schedule}]')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.grid(alpha=0.4)
    savefig(f'ddpm_{schedule}_loss')


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def test_cyclegan_face(weights, image_path):
    print(f"\nTesting CycleGAN with image: {image_path}")
    IMG_SIZE = 64
    G  = CycleGenerator().to(device)
    Fg = CycleGenerator().to(device)
    ckpt = torch.load(weights, map_location=device)
    G.load_state_dict(ckpt['G']); Fg.load_state_dict(ckpt['F'])
    G.eval(); Fg.eval()

    img_pil = Image.open(image_path).convert('RGB')
    short   = min(img_pil.size)
    transform = transforms.Compose([
        transforms.CenterCrop(short),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    img_t = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        to_blonde = G(img_t).squeeze(0).cpu()
        to_dark   = Fg(img_t).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, img in zip(
        axes,
        ['Original', 'G: → Blonde Hair', 'F: → Dark Hair'],
        [img_t.squeeze(0).cpu(), to_blonde, to_dark],
    ):
        ax.imshow(denorm(img).permute(1,2,0))
        ax.set_title(title, fontsize=12); ax.axis('off')
    plt.suptitle('CycleGAN — Face Translation', fontsize=14)
    plt.tight_layout()
    savefig('cyclegan_face_result')
    print("Result saved → figs/cyclegan_face_result.png")


def generate_ddpm(weights, n=64, schedule='linear'):
    print(f"\nGenerating {n} DDPM samples [{schedule} schedule] ...")
    betas, alphas, alpha_bar, sqrt_ab, sqrt_1mab = build_schedule(schedule)
    sqrt_recip = torch.sqrt(1.0 / alphas)
    post_var   = betas * (1.0 - F.pad(alpha_bar[:-1], (1,0), value=1.0)) / (1.0 - alpha_bar)

    unet = SimpleUNet().to(device)
    unet.load_state_dict(torch.load(weights, map_location=device))
    unet.eval()

    @torch.no_grad()
    def p_sample(x_t, t_scalar):
        t_b   = torch.full((x_t.size(0),), t_scalar, device=device, dtype=torch.long)
        pred  = unet(x_t, t_b)
        coeff = betas[t_scalar] / sqrt_1mab[t_scalar]
        mean  = sqrt_recip[t_scalar] * (x_t - coeff * pred)
        if t_scalar == 0: return mean
        return mean + torch.sqrt(post_var[t_scalar]) * torch.randn_like(x_t)

    x = torch.randn(n, 1, 28, 28, device=device)
    for t in tqdm(reversed(range(T)), total=T, desc='Sampling'):
        x = p_sample(x, t)

    grid = torchvision.utils.make_grid(x, nrow=8, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(10,10))
    plt.imshow(grid.permute(1,2,0).cpu()); plt.axis('off')
    plt.title(f'DDPM Generated Samples [{schedule}]')
    savefig(f'ddpm_{schedule}_samples')
    print(f"Saved → figs/ddpm_{schedule}_samples.png")


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='A4 Generative Models')
    p.add_argument('--model',      type=str, required=True,
                   choices=['gan', 'cyclegan', 'ddpm'],
                   help='Model to train or run')
    p.add_argument('--dataset',    type=str, default='mnist',
                   choices=['mnist', 'celeba'],
                   help='Dataset to use')
    p.add_argument('--epochs',     type=int, default=20,
                   help='Number of training epochs')
    p.add_argument('--train',      action='store_true',
                   help='Run training')
    p.add_argument('--weights',    type=str, default=None,
                   help='Path to saved weights for inference')
    p.add_argument('--test-image', type=str, default=None,
                   help='Path to face image for CycleGAN test')
    p.add_argument('--generate',   action='store_true',
                   help='Generate samples (DDPM)')
    p.add_argument('--n',          type=int, default=64,
                   help='Number of samples to generate')
    p.add_argument('--schedule',   type=str, default='linear',
                   choices=['linear', 'cosine'],
                   help='DDPM noise schedule')
    p.add_argument('--data-root',  type=str, default='./data',
                   help='Root directory for datasets')
    p.add_argument('--num-workers',type=int, default=4,
                   help='DataLoader num_workers')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Training ──────────────────────────────────────────────────────────────
    if args.train:
        if args.model == 'gan':
            train_gan(
                epochs=args.epochs,
                data_root=args.data_root,
                num_workers=args.num_workers,
            )
        elif args.model == 'cyclegan':
            train_cyclegan(
                epochs=args.epochs,
                data_root=args.data_root,
                num_workers=args.num_workers,
            )
        elif args.model == 'ddpm':
            train_ddpm(
                epochs=args.epochs,
                data_root=args.data_root,
                num_workers=args.num_workers,
                schedule=args.schedule,
            )

    # ── Inference ─────────────────────────────────────────────────────────────
    elif args.test_image and args.model == 'cyclegan':
        if not args.weights:
            print("ERROR: --weights required for --test-image")
            return
        test_cyclegan_face(args.weights, args.test_image)

    elif args.generate and args.model == 'ddpm':
        if not args.weights:
            print("ERROR: --weights required for --generate")
            return
        generate_ddpm(args.weights, n=args.n, schedule=args.schedule)

    else:
        print("ERROR: specify --train, --test-image, or --generate")
        print("Run with --help for usage.")


if __name__ == '__main__':
    main()
