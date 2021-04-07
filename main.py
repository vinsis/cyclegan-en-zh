import itertools
import argparse
import os

from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torch

from model import Generator, Discriminator
from data import dataset, CWD
from utils import weights_init_normal, LambdaLR
from loss import criterion_GAN, criterion_cycle, criterion_identity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
WEIGHTS_DIR = os.path.join(CWD, 'weights')

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--save_every', type=int, default=10)
opt = parser.parse_args()

BETAS = (0.5, 0.999)
DECAY_EPOCH = opt.n_epochs // 2

dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

netG_en2zh = Generator(3,3).to(device)
netG_zh2en = Generator(3,3).to(device)
netD_en = Discriminator(3).to(device)
netD_zh = Discriminator(3).to(device)

netG_en2zh.apply(weights_init_normal)
netG_zh2en.apply(weights_init_normal)
netD_en.apply(weights_init_normal)
netD_zh.apply(weights_init_normal)

# optimizers and learning rate schedulers
optimizer_G = Adam(itertools.chain(netG_en2zh.parameters(), netG_zh2en.parameters()), lr=opt.lr, betas=BETAS)
optimizer_D_en = Adam(netD_en.parameters(), lr=opt.lr, betas=BETAS)
optimizer_D_zh = Adam(netD_zh.parameters(), lr=opt.lr, betas=BETAS)

lr_scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda = LambdaLR(opt.n_epochs,0,DECAY_EPOCH).step)
lr_scheduler_D_en = lr_scheduler.LambdaLR(optimizer_D_en, lr_lambda = LambdaLR(opt.n_epochs,0,DECAY_EPOCH).step)
lr_scheduler_D_zh = lr_scheduler.LambdaLR(optimizer_D_zh, lr_lambda = LambdaLR(opt.n_epochs,0,DECAY_EPOCH).step)

def train():
    for epoch in range(opt.n_epochs):
        print('=== Starting epoch:', epoch, '===')
        lr_scheduler_G.step()
        lr_scheduler_D_en.step()
        lr_scheduler_D_zh.step()

        for index, data in enumerate(dataloader):
            real_data_en = data['en'].to(device)
            real_data_zh = data['zh'].to(device)

            ###################
            # Generator
            ###################

            # Identity loss
            recreated_real_data_zh = netG_en2zh(real_data_zh)
            recreated_real_data_en = netG_zh2en(real_data_en)
            identity_loss = criterion_identity(recreated_real_data_zh, real_data_zh) + criterion_identity(recreated_real_data_en, real_data_en)

            # Discriminator loss
            fake_data_zh = netG_en2zh(real_data_en)
            fake_data_en = netG_zh2en(real_data_zh)
            pred_fake_zh = netD_zh(fake_data_zh)
            pred_fake_en = netD_en(fake_data_en)
            discriminator_loss = criterion_GAN(pred_fake_zh, torch.ones_like(pred_fake_zh)) + criterion_GAN(pred_fake_en, torch.ones_like(pred_fake_en))

            # Cycle Consistency loss
            recycled_data_en = netG_zh2en(fake_data_zh)
            recycled_data_zh = netG_en2zh(fake_data_en)
            cycle_loss = criterion_cycle(recycled_data_en, real_data_en) + criterion_cycle(recycled_data_zh, real_data_zh)

            loss_G = 5 * identity_loss + discriminator_loss + 10 * cycle_loss
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            ###################
            # Discriminator
            ###################
            pred_real_zh = netD_zh(real_data_zh)
            pred_real_en = netD_en(real_data_en)

            pred_fake_zh = netD_zh(fake_data_zh.detach())
            pred_fake_en = netD_en(fake_data_en.detach())

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
            
            if index % 100 == 0:
                print('\nIndex:', index)
                print('\tLoss_G is', loss_G.item())
                print('\tLoss_D_en is', loss_D_en.item())
                print('\tLoss_D_zh is', loss_D_zh.item())

        if epoch % opt.save_every == 0:
            torch.save(netG_en2zh.state_dict(), os.path.join(WEIGHTS_DIR, 'netG_en2zh_epoch{}.pth'.format(epoch)))
            torch.save(netG_zh2en.state_dict(), os.path.join(WEIGHTS_DIR, 'netG_zh2en_epoch{}.pth'.format(epoch)))
            torch.save(netD_en.state_dict(), os.path.join(WEIGHTS_DIR, 'netD_en_epoch{}.pth'.format(epoch)))
            torch.save(netD_zh.state_dict(), os.path.join(WEIGHTS_DIR, 'netD_zh_epoch{}.pth'.format(epoch)))

if __name__ == '__main__':
    train()
