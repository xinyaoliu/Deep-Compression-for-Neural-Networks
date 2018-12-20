import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from models import LeNet_5
from models import LeNet_5_CIFAR
from huffmancoding import huffman_encode_model
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

parser = argparse.ArgumentParser(description='Deep Compression')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=66, 
                    help='random seed (default: 42)')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--datasets', type=str, default = 'mnist', help='choose different datasets')
parser.add_argument('--bits', type=float, default=4, 
                    help='number of clusters(2**bits)')
parser.add_argument('--output', default='saves/weight_sharing.ptmodel', type=str,
                    help='path to model output')

args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = not args.no_cuda
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA')

# dataloader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
if args.datasets == 'mnist':
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
            )),batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
            )),batch_size=1000, shuffle=False, **kwargs)

if args.datasets == 'cifar':
    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True,
                   transform=transforms.Compose(
                    [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
                       )),batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
            )),batch_size=1000, shuffle=False, **kwargs)



if args.datasets == 'mnist':
    model = LeNet_5(mask=True).to(device)
elif args.datasets == 'cifar':
    model = LeNet_5_CIFAR(mask=True).to(device)


optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs):
    train_loss = []
    test_acc = []
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            if args.datasets == 'cifar':
                data = torch.mean(data, dim=1).unsqueeze(1)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % 10 == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')
        train_loss.append(loss.item())
        test_acc.append(test())
    return train_loss, test_acc


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.datasets == 'cifar':
                data = torch.mean(data, dim=1).unsqueeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] 
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


# Initial training
print("--- Initial training ---")
start = time.time()
train_loss, test_acc = train(args.epochs)
end = time.time()
print ('running time:  ', str(end-start)+'s')
accuracy = test()
torch.save(model, f"saves/initial_model.ptmodel")
print("--- Before pruning ---")
print_nonzeros(model)

file1=open('results/'+str(args.datasets)+'_sensitivity'+str(args.sensitivity)+'__'+str(time.time())+'_init_loss.txt','w')
file1.write(str(train_loss));
file1.close()

file2=open('results/'+str(args.datasets)+'_sensitivity'+str(args.sensitivity)+'__'+str(time.time())+'_init_acc.txt','w')
file2.write(str(test_acc));
file2.close()

# Pruning
model.prune_by_std(args.sensitivity)

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) 
start = time.time()
train_loss, tets_acc = train(args.epochs)
end = time.time()
print ('running time:   ', str(end - start)+'s')
torch.save(model, f"saves/retraining.ptmodel")
accuracy = test()

print("--- After Retraining ---")
print_nonzeros(model)

file1=open('results/'+str(args.datasets)+'_sensitivity'+str(args.sensitivity)+'__'+str(time.time())+'_prune_retrain_loss.txt','w')
file1.write(str(train_loss));
file1.close()

file2=open('results/'+str(args.datasets)+'_sensitivity'+str(args.sensitivity)+'__'+str(time.time())+'_prune_retrain_acc.txt','w')
file2.write(str(test_acc));
file2.close()


def apply_weight_sharing(model, bits=4):
    # use the third method to initialize
    child_cnt = 0
    for module in model.children():
        child_cnt += 1
        if child_cnt < 4:
            continue
        #print (module)
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2**bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev)

def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    if args.datasets == 'mnist':
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=1000, shuffle=False, **kwargs)
    if args.datasets == 'cifar':
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=1000, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.datasets == 'cifar':
                data = torch.mean(data, dim = 1).unsqueeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

model = torch.load('saves/retraining.ptmodel')
print('ACC before weight sharing')
test(model, use_cuda)


apply_weight_sharing(model, bits=args.bits)
print('ACC after weight sharing')
test(model, use_cuda)
torch.save(model, args.output)


model = torch.load('saves/weight_sharing.ptmodel')
huffman_encode_model(model)
print('ACC after Huffman encoding')
test(model, use_cuda)

