import torch
from torch.cuda.amp import autocast
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

import numpy as np
import sys
import argparse
import copy

from datasets.GLDV2Dataset import *
from datasets.CUB200Dataset import *
from utils.model import *

batch_size = 256
img_size = 224
nesting_start=3

activation = {}
fwd_pass_x_list = []
fwd_pass_y_list = []
fwd_pass_paths_list = []

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, help='path to model to load from disk', type=str, required=True)
parser.add_argument('--dataset_name', default='cub200', help='dimension of last linear layer', type=str, required=True)
parser.add_argument('--distributed', default=0, help='is model DistributedDataParallel')
parser.add_argument('--gpu', default=0, help='available gpu device')
parser.add_argument('--world_size', default=1, help='distributed world size')
parser.add_argument('--nesting', default=0, help='was model trained with nesting', type=int)
parser.add_argument('--single_head', default=0, help='is model single head nested', type=int)
parser.add_argument('--fixed_feature', default=2048, help='dimension of last linear layer', type=int)
parser.add_argument('--workers', default=12, help='workers for dataloader', type=int)
parser.add_argument('--random_sample_dim', default=1281167, help='number of random samples to slice from val set', type=int)
args = parser.parse_args()

# Get the 2048 dim feature vector
def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

def append_fwd_pass_info_to_list(activation, label, paths):
    args = parser.parse_args()
    ff = args.fixed_feature

    #Iterate over batch and update X and y lists
    for i in range (activation.shape[0]):
        x = activation[i].cpu().detach().numpy()
        y = label[i].cpu().detach().numpy()
        fwd_pass_y_list.append(y)
        fwd_pass_x_list.append(x[:ff])
        #fwd_pass_paths_list.append(paths[i])

# save X (n x 2048), y (n x 1) and paths (n x 1) to disk, where n = num_samples
def dump_feature_vector(config_name, random_sample_dim):
        args = parser.parse_args()
        X_fwd_pass = np.asarray(fwd_pass_x_list, dtype=np.float32)
        y_fwd_pass = np.asarray(fwd_pass_y_list, dtype=np.float16).reshape(-1,1)
        #paths_fwd_pass = np.asarray(fwd_pass_paths_list)
        print("Feature Vector Array: ", X_fwd_pass.shape)
        print("Label Array: ", y_fwd_pass.shape)
        print("X filesize= %d, y filesize=%d" %(X_fwd_pass.size * X_fwd_pass.itemsize, y_fwd_pass.size * y_fwd_pass.itemsize))
        #print("Image Paths: ", len(fwd_pass_paths_list))

        if random_sample_dim < X_fwd_pass.shape[0]:
            random_indices = np.random.choice(X_fwd_pass.shape[0], size=random_sample_dim, replace=False)
            random_X = X_fwd_pass[random_indices, :]
            random_y = y_fwd_pass[random_indices, :]
            print("Writing random samples to disk with dim [%d x 2048] " % random_sample_dim)
        else:
            random_X = X_fwd_pass
            random_y = y_fwd_pass
            print("Writing %s to disk with dim [%d x %d]" % (str(config_name), X_fwd_pass.shape[0], args.fixed_feature))

        np.save('/mnt/disks/downstream/fwd_pass_csv/'+str(config_name)+'-X.npy', random_X)
        np.save('/mnt/disks/downstream/fwd_pass_csv/'+str(config_name)+'-y.npy', random_y)
        #np.savetxt('/mnt/disks/downstream/fwd_pass_csv/'+str(config_name)+'-X.csv', random_X, delimiter=",")
        #np.savetxt('/mnt/disks/downstream/fwd_pass_csv/'+str(config_name)+'-y.csv', random_y, delimiter=",")
        #np.savetxt('/mnt/disks/retrieval/nesting-retrieval/'+str(config_name)+'-paths.csv', paths_fwd_pass, fmt="%s", delimiter=",") 

def fwd_pass(data_loader, model, config, gpu, distributed, dataset, random_sample_dim):
    print("Fwd pass with random sample dim: " + str(random_sample_dim))
    model.eval()

    if distributed:
        model.module.avgpool.register_forward_hook(get_activation('avgpool'))
    else:
        model.avgpool.register_forward_hook(get_activation('avgpool'))

    with torch.no_grad():
        with autocast():
                for i_batch, sample_batched in enumerate(data_loader):
                    if dataset=='cub200':
                        images, target, paths  = sample_batched['image'], sample_batched['cls'], sample_batched['path']
                        print(i_batch, images.shape, target[100])
                    elif dataset=='imagenet1k':
                        images, target = sample_batched
                        paths=None

                    images, target = images.to(gpu), target.to(gpu)
                    output = model(images)
                    append_fwd_pass_info_to_list(activation['avgpool'].squeeze(), target, paths)
                    if i_batch % 50 == 0:
                        print("Finished processing: %f %%" % (i_batch / len(data_loader) * 100))
                dump_feature_vector(config, random_sample_dim)

def main():

    args = parser.parse_args()

    # replace normalize with cub mean+std for cub fwd pass
    cub_mean = (0.485, 0.499, 0.4432)
    cub_std = (0.232, 0.227, 0.265)

    #default: imagenet mean+std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        normalize,
    ])

    # Not used for retrieval, DB and query should have same distribution
    '''
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])
    '''

    model = Model(args.distributed, args.gpu, args.nesting, args.single_head, args.fixed_feature, use_blurpool=1)
    temp = copy.copy(model)

    if args.distributed:
        model.setup_distributed(args.world_size)
        print("setup distributed")

    model_init = model.initModel()
    model_weights_disk = args.model_path
    model = model.load_model(model_init, model_weights_disk)
    print("Loaded pretrained model: " + str(model_weights_disk))
    #print("Pretrained model fc layer: ", model.module.fc)

    torch.cuda.set_device(args.gpu)
    device = f'cuda{args.gpu}'

    if(args.dataset_name =='cub200'):
        root = '/mnt/disks/retrieval/nesting-retrieval/cub/data/CUB_200_2011/images'
        train_file = 'datasets/cub200_train.csv'
        test_file = 'datasets/cub200_test.csv'
        train_dataset = CUB200Dataset(csv_file=train_file, root_dir=root_dir,transform=test_transform)
        test_dataset = CUB200Dataset(csv_file=test_file, root_dir=root_dir,transform=test_transform)
        num_classes = 200
    elif(args.dataset_name =='gldv2'):
        root = '/mnt/disks/retrieval/nesting-retrieval/google-landmark/'
        train_file = str(root)+'csv/gldv2-database-master.csv'
        test_file = str(root)+'csv/gldv2-query-master.csv'
    elif(args.dataset_name == 'imagenet1k'):
        train_path = '/home/jupyter/imagenet/train/'
        test_path =  '/home/jupyter/imagenet/val/'
        train_dataset = datasets.ImageFolder(train_path, transform = test_transform)
        test_dataset = datasets.ImageFolder(test_path, transform = test_transform)
        num_classes = 1000
    else:
        print("Error: unsupported dataset!")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #run fwd pass on train dataset
    config = args.dataset_name+"_val_nesting"+str(args.nesting)+"_sh"+str(args.single_head)+"_ff"+str(args.fixed_feature) ; print("Config: " + config)
    fwd_pass(val_loader, model, config, args.gpu, args.distributed, args.dataset_name, args.random_sample_dim)

    # re-initialize lists for test dataset
    global fwd_pass_x_list
    global fwd_pass_y_list
    global fwd_pass_paths_list
    fwd_pass_x_list = []
    fwd_pass_y_list = []
    fwd_pass_paths_list = []

    config = args.dataset_name+"_train_nesting"+str(args.nesting)+"_sh"+str(args.single_head)+"_ff"+str(args.fixed_feature) ; print("Config: " + config)
    fwd_pass(train_loader, model, config, args.gpu, args.distributed, args.dataset_name, args.random_sample_dim)

    if args.distributed:
        temp.cleanup_distributed()

if __name__ == "__main__":
    main()
