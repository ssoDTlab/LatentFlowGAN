import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from cleanfid import fid
import lpips
import torch.nn.functional as F
from torchvision import models

def plot_single_metric(metric_scores, metric_name, save_path):
    epochs = [ep for ep, _ in metric_scores]
    values = [score for _, score in metric_scores]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_lpips_near_far(lpips_scores, save_path):
    epochs = [ep for ep, _, _ in lpips_scores]
    lpips_near = [near for _, near, _ in lpips_scores]
    lpips_far = [far for _, _, far in lpips_scores]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lpips_near, label="LPIPS (near)", marker="o")
    plt.plot(epochs, lpips_far, label="LPIPS (far)", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("LPIPS")
    plt.title("LPIPS (near vs. far) over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_gan(epoch_setting, dataset_name):
    print(f"\n{'=' * 50}")
    print(f"에폭 {epoch_setting}으로 학습 시작")
    print(f"{'=' * 50}\n")

    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    if dataset_name == "lsun_bedroom":
        dataroot = "/content/data/lsun_bedroom/data0/lsun/bedroom"
    elif dataset_name == "celebA":
        dataroot = "/content/data/celeba"
    elif dataset_name == "ffhq":
        dataroot = "/content/data/ffhq"
    elif dataset_name == "ImageNet64":
        dataroot = "/content/data/ImageNet64"

    workers = 8 
    
    batch_size = 128

    image_size = 64

    nc = 3

    nz = 100

    ngf = 64

    ndf = 64

    num_epochs = epoch_setting

    lr_D = 0.0002
    lr_G = 0.0002
    beta1 = 0.5

    ngpu = 1

    # d(z1,z2) = tau , d(z1,z2) = eps
    eps = 0.1
    tau = 10.0

   
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    class PerceptualCosineLoss(nn.Module):
        def __init__(self, layer_idx=16):
            super().__init__()
            vgg = models.vgg16(pretrained=True).features
            self.extractor = nn.Sequential(*list(vgg.children())[:layer_idx]).eval()
            for p in self.extractor.parameters():
                p.requires_grad = False

        def forward(self, fake_batch):
            B = fake_batch.size(0)

            fake_norm = (fake_batch + 1) / 2

            with torch.no_grad():  
                features = self.extractor(fake_norm)

            features_flat = features.view(B, -1)
            features_norm = F.normalize(features_flat, p=2, dim=1)

            similarity_matrix = torch.mm(features_norm, features_norm.t())

            mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
            avg_similarity = similarity_matrix[mask].mean()

            return avg_similarity

    # Generator Code
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. ``(ngf*8) x 4 x 4``
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. ``(ngf*4) x 8 x 8``
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. ``(ngf*2) x 16 x 16``
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. ``(ngf) x 32 x 32``
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. ``(nc) x 64 x 64``
            )

        def forward(self, input):
            return self.main(input)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is ``(nc) x 64 x 64``
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf) x 32 x 32``
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*2) x 16 x 16``
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*4) x 8 x 8``
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*8) x 4 x 4``
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input).view(-1)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD)

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    perceptual_cosine_loss = PerceptualCosineLoss().to(device)

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

    d_losses = []
    g_losses = []
    div_losses = []
    fid_scores = []
    lpips_scores = []

    # 학습 과정
    print("Starting Training Loop...")
    for epoch in range(num_epochs):

        for i, data in enumerate(dataloader, 0):

           
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output_real = netD(real_cpu)
            errD_real = criterion(output_real, label_real)
            D_x = output_real.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_img = netG(noise)

            output_fake = netD(fake_img.detach())
            errD_fake = criterion(output_fake, label_fake)
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

           
            netG.zero_grad()

            output_fake_G = netD(fake_img)
            errG_adv = criterion(output_fake_G, label_real)

            errG_div = perceptual_cosine_loss(fake_img)  
            lambda_div = 1.0

            errG = errG_adv + lambda_div * errG_div

            errG.backward()
            D_G_z2 = output_fake_G.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                with torch.no_grad():
                    d_losses.append(errD.item())
                    g_losses.append(errG.item())
                    div_losses.append(errG_div.item())
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G_adv: %.4f\tLoss_G_div: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                       errD.item(), errG_adv.item(), errG_div.item(), D_x, D_G_z1, D_G_z2))

        if (epoch + 1) % 10 == 0:
            # Create two subplots side by side
            plt.figure(figsize=(15, 5))

            # First subplot - Adversarial losses
            plt.subplot(1, 3, 1)
            plt.title(f"Adversarial Losses (Epoch {epoch + 1})")
            plt.plot(g_losses, label="Generator")
            plt.plot(d_losses, label="Discriminator")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            # Second subplot - Diversity loss
            plt.subplot(1, 3, 2)
            plt.title(f"Diversity Loss (Epoch {epoch + 1})")
            plt.plot(div_losses, label="Diversity Loss", color="orange")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            # Third subplot - Combined view
            plt.subplot(1, 3, 3)
            plt.title(f"All Losses (Epoch {epoch + 1})")
            plt.plot(g_losses, label="Generator", alpha=0.7)
            plt.plot(d_losses, label="Discriminator", alpha=0.7)
            plt.plot(div_losses, label="Diversity", alpha=0.7)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"/content/losses_epoch_{epoch + 1}.png")
            plt.close()

            with torch.no_grad():
                sample_noise = torch.randn(64, nz, 1, 1, device=device)
                sample_images = netG(sample_noise)
                vutils.save_image(sample_images.detach(),
                                  f"/content/samples_epoch_{epoch + 1}.png",
                                  normalize=True,
                                  nrow=8)

        if (epoch + 1) % 10 == 0:
            # LPIPS_near, LPIPS_far
            z1 = torch.randn(batch_size, nz, 1, 1, device=device)
            u = torch.randn_like(z1)
            u = u / u.norm(dim=1, keepdim=True)
            z2_near = z1 + eps * u
            z2_far = z1 + tau * u

            img_z1 = (netG(z1) + 1) / 2
            img_near = (netG(z2_near) + 1) / 2
            img_far = (netG(z2_far) + 1) / 2

            lpips_near = lpips_fn(img_z1, img_near).mean().item()
            lpips_far = lpips_fn(img_z1, img_far).mean().item()

           
            real_dir = f"/content/GAN_epoch_{epoch + 1}_real_images"
            fake_dir = f"/content/GAN_epoch_{epoch + 1}_fake_images"
            os.makedirs(real_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)

            num_samples = 50000

            num_full_batches = num_samples // batch_size
            remaining_samples = num_samples % batch_size

            real_counter = 0
            fake_counter = 0

            with torch.no_grad():
                data_iter = iter(dataloader)
                for i in range(num_full_batches):
                    try:
                        real_batch = next(data_iter)[0].to(device)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        real_batch = next(data_iter)[0].to(device)

                    for j in range(real_batch.size(0)):
                        vutils.save_image(real_batch[j], f"{real_dir}/img_{real_counter}.png", normalize=True)
                        real_counter += 1

                    noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake_batch = netG(noise)
                    for j in range(fake_batch.size(0)):
                        vutils.save_image(fake_batch[j], f"{fake_dir}/img_{fake_counter}.png", normalize=True)
                        fake_counter += 1

                if remaining_samples > 0:
                    real_batch = next(data_iter)[0][:remaining_samples].to(device)
                    for j in range(real_batch.size(0)):
                        vutils.save_image(real_batch[j], f"{real_dir}/img_{real_counter}.png", normalize=True)
                        real_counter += 1

                    noise = torch.randn(remaining_samples, nz, 1, 1, device=device)
                    fake_batch = netG(noise)
                    for j in range(fake_batch.size(0)):
                        vutils.save_image(fake_batch[j], f"{fake_dir}/img_{fake_counter}.png", normalize=True)
                        fake_counter += 1

            fid_score = fid.compute_fid(real_dir, fake_dir, mode="clean", dataset_res=64, num_workers=0)


            fid_scores.append((epoch + 1, fid_score))
            lpips_scores.append((epoch + 1, lpips_near, lpips_far))

            score_path = "/content/fid_lpips_scores.txt"
            if not os.path.exists(score_path):
                with open(score_path, 'w') as f:
                    f.write("Epoch\tFID\tLPIPS_near\tLPIPS_far\n")
            with open(score_path, 'a') as f:
                f.write(f"{epoch + 1}\t{fid_score:.4f}\t{lpips_near:.4f}\t{lpips_far:.4f}\n")

            plot_single_metric(fid_scores, "FID", "/content/fid_plot.png")
            plot_lpips_near_far(lpips_scores, "/content/lpips_plot.png")


def main():
    epoch = 300
    train_gan(epoch, "celebA")


if __name__ == '__main__':
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
