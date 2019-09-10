import numpy as np
import time,os
from argparse import ArgumentParser
import torch
import torchvision
import umap
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.utils.data as utils
from tqdm import tqdm
from pyutils import population_mean_norm,show
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
NETWORK DEFINITIONS

'''

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(nn.Linear(32,2))
    self.linears = nn.Sequential(
        nn.Linear(512*2*2, 512),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, 32),
        nn.Dropout(0.3),
        nn.ReLU()
    )
    
  def forward(self, z_s):
    batch_size = z_s.shape[0]
    z_s = z_s.view(batch_size,512*2*2)
    maps = self.linears(z_s)
    preds = self.layer1(maps)
    preds = nn.Sigmoid()(preds)
    return maps,preds
    
class FaderNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))



    self.conv1 = nn.Sequential(
        nn.ConvTranspose2d(512+2, 512, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv2 = nn.Sequential(
        nn.ConvTranspose2d(512+2, 256, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv3 = nn.Sequential(
        nn.ConvTranspose2d(256+2, 128, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv4 = nn.Sequential(
        nn.ConvTranspose2d(128+2, 64, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv5 = nn.Sequential(
        nn.ConvTranspose2d(64+64+2, 32, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv6 = nn.Sequential(
        nn.ConvTranspose2d(32+32+2, 16, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv7 = nn.Sequential(
        nn.ConvTranspose2d(16+16+2, 3, 4, 2, 1, bias=False),
        nn.ReLU()
        )

      
  def forward(self, imgs, labels):
    batch_size = imgs.shape[0]

    z_s,skip1,skip2,skip3 = self.encode(imgs)
    
    reconsts = self.decode(z_s, labels,skip1,skip2,skip3)
    
    return reconsts
 
  def encode(self, imgs):
    out1 = self.layer1(imgs)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)

    return out7,out1,out2,out3
  
  def decode_prob(self, z_s, hot_labels,skip1,skip2,skip3):
    z_s = torch.cat([z_s, hot_labels], dim=1)

    out1 = self.conv1(z_s)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2) #expand the label vector to concatenate with intermediate outputs
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    out1 = torch.cat([out1, hot_labels], dim=1)
    
    out2 = self.conv2(out1)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    out2 = torch.cat([out2, hot_labels], dim=1)
    
    out3 = self.conv3(out2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    out3 = torch.cat([out3, hot_labels], dim=1)

    out4 = self.conv4(out3)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    # out4 = torch.cat([out4, hot_labels], dim=1)
    out4 = torch.cat([out4, skip3], dim=1)
    out4 = torch.cat([out4, hot_labels], dim=1)

    out5 = self.conv5(out4)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    # out5 = torch.cat([out5, hot_labels], dim=1)
    out5 = torch.cat([out5, skip2], dim=1)
    out5 = torch.cat([out5, hot_labels], dim=1)

    out6 = self.conv6(out5)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    # out6 = torch.cat([out6, hot_labels], dim=1)
    out6 = torch.cat([out6, skip1], dim=1)
    out6 = torch.cat([out6, hot_labels], dim=1)

    out7 = self.conv7(out6)

    return out7
  def decode(self, z_s, labels,skip1,skip2,skip3):
    batch_size = len(labels)
    hot_digits = torch.zeros((batch_size, 2, 2, 2)).to(device)
    labels = labels.long()
    for i, digit in enumerate(labels):
      hot_digits[i,digit,:,:] = 1
    
    return self.decode_prob(z_s, hot_digits,skip1,skip2,skip3)






'''
TRAIN/TEST

'''






parser = ArgumentParser(description = "customize training")
parser.add_argument('--disc_schedule', '-ds', default = '0.000001')
parser.add_argument('--fader_lr', '-f', default = '0.0002')
parser.add_argument('--disc_lr', '-d', default = '0.0002')
args = parser.parse_args()

pop_mean, pop_std0 = population_mean_norm(path = "../select")

train_dataset = torchvision.datasets.ImageFolder(
        root="../select",

        transform=transforms.Compose([
                # transforms.RandomCrop((178,178)),
                # transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=pop_mean, std=pop_std0)
                ])

    )

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32, 
        num_workers=0,
        shuffle=True
    )

fader = FaderNetwork().to(device)
disc = Discriminator().to(device)
fader.cuda()
disc.cuda()

# print(fader)


#TRAIN/TEST


umap_data1 = []
umap_data2 = []
umap_labels = []
disc_data = []

fader_optim = optim.Adam(fader.parameters(), lr=float(args.fader_lr), betas=(0.5,0.999))
disc_optim = optim.Adam(disc.parameters(), lr=float(args.disc_lr), betas=(0.5,0.999))

def train(epoch):
    fader.train()
    disc.train()

    sum_disc_loss = 0
    sum_disc_acc = 0
    
    sum_rec_loss = 0
    
    sum_fader_loss = 0
    
    disc_weight = 0

    disc_weight = epoch*float(args.disc_schedule)


    for data, labels in tqdm(train_loader, desc="Epoch {}".format(epoch)):
        data = data.to(device)
        labels = labels.long().to(device)
        
        # Encode data
        z,skip1,skip2,skip3 = fader.encode(data)
        # print('z is ',z[0,])

        if epoch%10 == 0 and z.shape[0] == 32:
            z_temp = z.view(z.shape[0], 512*2*2)
            z_temp = z_temp.cpu()
            temp = z_temp.detach().numpy() 
            labs = labels.cpu().detach().numpy()
            # standard_embedding = umap.UMAP(random_state=42).fit_transform(temp)
            umap_data1.append(temp[1]) 
            umap_labels.append(labs[1])
            # plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labs, s=0.1, cmap='Spectral')
            # plt.savefig('results/umapresults'+str(epoch)+'.png')

        
        # Train discriminator
        disc_optim.zero_grad()        
        maps,label_probs = disc(z)
        # print('disc ', label_probs[0,])
        disc_loss = F.cross_entropy(label_probs, labels, reduction='mean')
        sum_disc_loss += disc_loss.item()
        disc_loss.backward()
        disc_optim.step()

        if epoch%10 == 0 and z.shape[0] == 32:
            temp = maps.cpu().detach().numpy() 
            disc_data.append(temp[1]) 
            # plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labs, s=0.1, cmap='Spectral')
            # plt.savefig('results/umapresults'+str(epoch)+'.png')

        # Compute discriminator accuracy
        disc_pred = torch.argmax(label_probs, 1)
        disc_acc = torch.sum(disc_pred == labels)
        sum_disc_acc += disc_acc.item()
        
        
        # Train Fader
        fader_optim.zero_grad()
        z,skip1,skip2,skip3 = fader.encode(data)
        
        # Invariance of latent space from new disc
        _,label_probs = disc(z)
        
        # Reconstruction
        reconsts = fader.decode(z, labels,skip1,skip2,skip3)
        # print('reconstructions ',)
        rec_loss = F.mse_loss(reconsts, data, reduction='mean')
        sum_rec_loss += rec_loss.item()
        
        # if epoch > 50:
        #     fader_loss = rec_loss
        fader_loss = rec_loss - disc_weight * F.cross_entropy(label_probs, labels, reduction='mean')
        # fader_loss = rec_loss - disc_weight * F.cross_entropy(label_probs, labels, reduction='sum')
        # fader_loss = F.cross_entropy(label_probs, labels, reduction='sum')


        fader_loss.backward()
        fader_optim.step()
        
        sum_fader_loss += fader_loss.item()        
        
    train_size = len(train_loader.dataset)


    if epoch%10 == 0:
        plt.clf()
        # umap_data1 = np.reshape(umap_data1, (umap_data1[0]*umap_data1[1],umap_data1[2]))
        standard_embedding = umap.UMAP(random_state=42).fit_transform(umap_data1)
        plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=umap_labels, s=0.1, cmap='Spectral')
        plt.savefig('results/umapresults_fader'+str(epoch)+'.png')
        plt.clf()
        standard_embedding = umap.UMAP(random_state=42).fit_transform(disc_data)
        plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=umap_labels, s=0.1, cmap='Spectral')
        plt.savefig('results/umapresults_disc'+str(epoch)+'.png')
        umap_data1.clear()
        umap_labels.clear()
        disc_data.clear()


    print('\nDisc Weight: {:.8f} | Fader Loss: {:.8f} | Rec Loss: {:.8f} | Disc Loss, Acc: {:.8}, {:.8f}'
          .format(disc_weight, sum_fader_loss/train_size, sum_rec_loss/train_size, 
        sum_disc_loss/train_size, sum_disc_acc/train_size), flush=True)
    
    return sum_rec_loss/train_size, sum_disc_acc/train_size, disc_weight

def test(epoch):
    fader.eval()
    disc.eval()
    rec_losses = 0
    disc_losses = 0
    disc_accs = 0
    rec_accs = 0
    flag = 0
    with torch.no_grad():
        for data_batch, labels in train_loader:
            labels = labels.long().to(device)   
            data_batch = data_batch.to(device)
            z,skip1,skip2,skip3 = fader.encode(data_batch)

            _,label_probs = disc(z)
            disc_loss = F.cross_entropy(label_probs, labels, reduction='mean')
            reconsts = fader(data_batch, labels)
            rec_loss = F.mse_loss(reconsts, data_batch, reduction='mean')
            disc_pred = torch.argmax(label_probs, 1)
            disc_acc = torch.sum(disc_pred == labels)   

            disc_losses += disc_loss.item()
            rec_losses += rec_loss.item()
            disc_accs += disc_acc.item()

            if data_batch.shape[0] < 32:
                flag = 1
          
            data_batch = data_batch[:1].to(device)
            labels = labels[:1].to(device)

            batch_z,skip1,skip2,skip3= fader.encode(data_batch)

            # print(batch_z)

            con1 = torch.cat((batch_z, batch_z), 0)
            skip11 = torch.cat([skip1, skip1], 0) #pass the skip connections over to the decoder
            skip22 = torch.cat([skip2, skip2], 0)
            skip33 = torch.cat([skip3, skip3], 0)
            con2 = torch.cat((con1, batch_z), 0)
            con3 = torch.cat((con2, batch_z), 0)

            faders = (torch.tensor([0,1]).long()).to(device)

            '''
            KEYS

            0 - original
            1 - reconstruction with original attributes
            2 - reconstruction with modified attributes

            '''

            # if flag == 1:

            plt.clf()

            show(make_grid(data_batch.detach().cpu()), 'Epoch {} Original'.format(epoch),epoch,0)

            reconst = fader.decode(batch_z,labels,skip1,skip2,skip3).cpu()
            # print("*********************RECONST*************************\n",reconst)
            show(make_grid(reconst.view(1, 3, 256, 256)), 'Epoch {} Reconst with Orig Attr'.format(epoch),epoch,1)

            fader_reconst = fader.decode(con1,faders,skip11,skip22,skip33).cpu()
            show(make_grid(fader_reconst.view(2, 3, 256, 256), nrow=2), 'Epoch {} Reconst With Attr 0,3'.format(epoch),epoch,2)
            break

        print('Test Rec Loss: {:.8f}'.format(rec_losses / len(train_loader.dataset)))
        print('Test disc Loss: {:.8f}'.format(disc_losses / len(train_loader.dataset)))
        print('Test disc accs: {:.8f}'.format(disc_accs / len(train_loader.dataset)))

epochs = 1001 

recs, accs, disc_wts = [], [], []
for epoch in range(epochs):
    rec_loss, disc_acc, disc_wt = train(epoch)
    recs.append(rec_loss)
    accs.append(disc_acc)
    disc_wts.append(disc_wt)
    
    if epoch % 50 == 0:
        test(epoch)
        plt.figure(figsize=(9,3))
        plt.subplot(1,3,1)
        plt.title('Disc Weight')
        plt.plot(disc_wts)
        plt.subplot(1,3,2)
        plt.title('Reconst Loss')
        plt.plot(recs)
        plt.subplot(1,3,3)
        plt.title('Disc Acc')
        plt.plot(accs)
        plt.savefig('results/losses'+str(epoch)+'.png')
        torch.save(fader.state_dict(), 'results/fader'+str(epoch)+'.pt')


plt.figure(figsize=(9,3))
plt.subplot(1,3,1)
plt.title('Disc Weight')
plt.plot(disc_wts)
plt.subplot(1,3,2)
plt.title('Reconst Loss')
plt.plot(recs)
plt.subplot(1,3,3)
plt.title('Disc Acc')
plt.plot(accs)

# plt.savefig('losses1.png')

# """# Fade"""

# fader.load_state_dict(torch.load('fader.pt'))

# import random
# import numpy as np
# fader.eval()
# for imgs, labels in train_loader:
#   labels = labels.long()  
#   with torch.no_grad():
#     an_img = imgs[0].to(device)
#     label = labels[0].item()

#     z = fader.encode(an_img.unsqueeze(0).to(device))

#     z_s = z.repeat(8,1)

#     target = None
#     while target is None or target == label:
#       target = random.randint(0,3)

#     hot_start = np.zeros(2)
#     hot_start[label] = 1

#     hot_target = np.zeros(2)
#     hot_target[target] = 1

#     interp = np.linspace(hot_start, hot_target, num=8)

#     interp = torch.from_numpy(interp).float().to(device)

#     fades = fader.decode_prob(z_s, interp)

#     show(make_grid(fades.cpu(), nrow=8),"{} to {}".format(label, target),999,999)

#     break

