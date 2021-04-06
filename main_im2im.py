import itertools
import argparse
import os

from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torch

from model import Generator, Discriminator
from data_im2im import ImagePairDataset, CWD
from utils import weights_init_normal, LambdaLR
from loss import criterion_GAN, criterion_cycle, criterion_identity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
WEIGHTS_DIR = os.path.join(CWD, 'weights')



def train():
    for epoch in range(opt.n_epochs):
        print('=== Starting epoch:', epoch, '===')
        lr_scheduler_G.step()
        lr_scheduler_D_en.step()
        lr_scheduler_D_zh.step()

        for index, data in enumerate(dataloader):
            real_data_en = data['A'].to(device)
            real_data_zh = data['B'].to(device)

            ###################
            # Generator
            ###################

            # Identity loss
            recreated_real_data_zh = netG_A2B(real_data_zh)
            recreated_real_data_en = netG_B2A(real_data_en)
            identity_loss = criterion_identity(recreated_real_data_zh, real_data_zh) + criterion_identity(recreated_real_data_en, real_data_en)

            # Discriminator loss
            fake_data_zh = netG_A2B(real_data_en)
            fake_data_en = netG_B2A(real_data_zh)
            pred_fake_zh = netD_B(fake_data_zh)
            pred_fake_en = netD_A(fake_data_en)
            discriminator_loss = criterion_GAN(pred_fake_zh, torch.ones_like(pred_fake_zh)) + criterion_GAN(pred_fake_en, torch.ones_like(pred_fake_en))

            # Cycle Consistency loss
            recycled_data_en = netG_B2A(fake_data_zh)
            recycled_data_zh = netG_A2B(fake_data_en)
            cycle_loss = criterion_cycle(recycled_data_en, real_data_en) + criterion_cycle(recycled_data_zh, real_data_zh)

            loss_G = 5 * identity_loss + discriminator_loss + 10 * cycle_loss
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            ###################
            # Discriminator
            ###################
            pred_real_zh = netD_B(real_data_zh)
            pred_real_en = netD_A(real_data_en)

            pred_fake_zh = netD_B(fake_data_zh.detach())
            pred_fake_en = netD_A(fake_data_en.detach())

            loss_D_zh = criterion_GAN(pred_real_zh, torch.ones_like(pred_real_zh)) + criterion_GAN(pred_fake_zh, torch.zeros_like(pred_fake_zh))
            loss_D_en = criterion_GAN(pred_real_en, torch.ones_like(pred_real_en)) + criterion_GAN(pred_fake_en, torch.zeros_like(pred_fake_en))
            loss_D_zh = 0.5 * loss_D_zh
            loss_D_en = 0.5 * loss_D_en

            optimizer_D_zh.zero_grad()
            loss_D_zh.backward()
            optimizer_D_zh.step()

            optimizer_D_en.zero_grad()
            loss_D_en.backward()
            optimizer_D_en.step()
            
            if index % 10 == 0:
                print('\nIndex:', index)
                print('\tLoss_G is', loss_G.item())
                print('\tLoss_D_A is', loss_D_en.item())
                print('\tLoss_D_B is', loss_D_zh.item())

        if epoch % opt.save_every == 0 or epoch == opt.n_epochs-1:
            torch.save(netG_A2B.state_dict(), os.path.join(WEIGHTS_DIR, 'netG_A2B_epoch{}.pth'.format(epoch)))
            torch.save(netG_B2A.state_dict(), os.path.join(WEIGHTS_DIR, 'netG_B2A_epoch{}.pth'.format(epoch)))
            torch.save(netD_A.state_dict(), os.path.join(WEIGHTS_DIR, 'netD_A_epoch{}.pth'.format(epoch)))
            torch.save(netD_B.state_dict(), os.path.join(WEIGHTS_DIR, 'netD_B_epoch{}.pth'.format(epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--dirA', required=True)
    parser.add_argument('--dirB', required=True)
    opt = parser.parse_args()

    BETAS = (0.5, 0.999)
    DECAY_EPOCH = opt.n_epochs // 2

    dataset = ImagePairDataset(opt.dirA, opt.dirB)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    netG_A2B = Generator(3,3).to(device)
    netG_B2A = Generator(3,3).to(device)
    netD_A = Discriminator(3).to(device)
    netD_B = Discriminator(3).to(device)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # optimizers and learning rate schedulers
    optimizer_G = Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr, betas=BETAS)
    optimizer_D_en = Adam(netD_A.parameters(), lr=opt.lr, betas=BETAS)
    optimizer_D_zh = Adam(netD_B.parameters(), lr=opt.lr, betas=BETAS)

    lr_scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda = LambdaLR(opt.n_epochs,0,DECAY_EPOCH).step)
    lr_scheduler_D_en = lr_scheduler.LambdaLR(optimizer_D_en, lr_lambda = LambdaLR(opt.n_epochs,0,DECAY_EPOCH).step)
    lr_scheduler_D_zh = lr_scheduler.LambdaLR(optimizer_D_zh, lr_lambda = LambdaLR(opt.n_epochs,0,DECAY_EPOCH).step)

    train()
