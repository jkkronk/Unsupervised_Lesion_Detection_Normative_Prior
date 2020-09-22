__author__ = 'jonatank'
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import argparse
import yaml

from model.ConvVAE  import ConvVAE, train_vae, valid_vae
from dataloader import camcan_dataset

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument("--config", required=True, help="path to config")

    opt = parser.parse_args()

    model_name = opt.model_name

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    path = str(config['path'])
    epochs = int(config['epochs'])
    batch_size = int(config["batch_size"])
    img_size = int(config["spatial_size"])
    lr_rate = float(config['lr_rate'])
    log_freq = int(config['log_freq'])
    log_dir = str(config['log_dir'])

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load data
    print("Loading data...")
    train_dataset = camcan_dataset(path, True, img_size, data_aug=1)
    train_data_loader  = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Train data loaded')

    validation_dataset = camcan_dataset(path, False, img_size)
    valid_data_loader  = data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Valid data loaded')

    # Create model
    vae_model = ConvVAE(img_size, model_name)
    vae_model.double().to(device)

    #print(vae_model)

    # Init Optimizer Adam
    optimizer = optim.Adam(vae_model.parameters(), lr=lr_rate)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr_rate//100)

    # Init logging with Tensorboard
    writer_train = SummaryWriter(log_dir + '/train_' + vae_model.name)
    writer_valid = SummaryWriter(log_dir + '/valid_' + vae_model.name)

    # Start training
    print('Start training:')
    for epoch in range(epochs):
        print('Epoch:', epoch)
        # Train loss
        loss, lat_loss, l2_loss = train_vae(vae_model, train_data_loader, device, optimizer, epoch)
        # Validation loss
        valid_loss, valid_lat_loss, valid_l2_loss = valid_vae(vae_model, valid_data_loader, device, epoch)

        # Cosine annealing
        #scheduler.step()

        print(("epoch %d: train_l2_loss %f train_lat_loss %f total train_loss %f") % (
                    epoch, l2_loss, lat_loss, loss))

        print(("epoch %d: test_l2_loss %f test_lat_loss %f total loss %f") % (
            epoch, valid_l2_loss, valid_lat_loss, valid_loss))

        # Write to Tensorboard
        writer_train.add_scalar('total loss', loss, epoch)
        writer_train.add_scalar('l2 loss', l2_loss, epoch)
        writer_train.add_scalar('latent loss', lat_loss, epoch)

        writer_valid.add_scalar('total loss', valid_loss, epoch)
        writer_valid.add_scalar('l2 loss', valid_l2_loss, epoch)
        writer_valid.add_scalar('latent loss', valid_lat_loss, epoch)

        writer_train.flush()
        writer_valid.flush()

        # Save model
        if epoch % log_freq == 0: #and not epoch == 0:
            vae_model.eval()
            lat_batch_sample = vae_model.sample(batch_size, device)
            writer_valid.add_image('Batch of sampled images', torch.clamp(lat_batch_sample, 0, 1), epoch, dataformats='NCHW')

            img_test, mask = next(iter(valid_data_loader))
            img_test = img_test.to(device)
            img_re, __, __ = vae_model(img_test.double())
            writer_valid.add_image('Batch of original images', img_test, epoch, dataformats='NCHW')
            writer_valid.add_image('Batch of reconstructed images', torch.clamp(img_re, 0, 1), epoch, dataformats='NCHW')

            path = log_dir + model_name + str(epoch) + '.pth'
            torch.save(vae_model, path)


