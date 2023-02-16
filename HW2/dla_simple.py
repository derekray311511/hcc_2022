'''Simplified version of DLA in PyTorch.

Note this implementation is not identical to the original paper version.
But it seems works fine.

See dla.py for the original paper version.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.root = Root(2*out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels,
                                  out_channels, level=level-1, stride=stride)
            self.right_tree = Tree(block, out_channels,
                                   out_channels, level=level-1, stride=1)

    def forward(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root([out1, out2])
        return out


class SimpleDLA(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(SimpleDLA, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = SimpleDLA()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())


from torchvision import datasets ,transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np
# from google.colab.patches import cv2_imshow
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
# from tensorflow import summary
from torchvision.utils import make_grid
import os, cv2, argparse
import torch.backends.cudnn as cudnn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    EPOCH = 200                # 全部data訓練10次
    BATCH_SIZE = 128           # 每次訓練隨機丟50張圖像進去
    LR = args.lr              # learning rate
    # input_shape = (1,3,48,48)

    # #Load data set
    # print('==> Preparing data..')
    # train_data = datasets.ImageFolder(
    #     'CIFAR10/train',
    #     transform = transforms.Compose([transforms.RandomCrop(32, padding=4), 
    #                                     transforms.RandomHorizontalFlip(), 
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #                                     ])                         
    # )

    # test_data = datasets.ImageFolder(
    #     'CIFAR10/test',
    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #                                     ])                         
    # )
    # # Pytorch DataLoader
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle= True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=100,shuffle=False, num_workers=2)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    #show labels
    print("Classes:", train_data.classes)
    print("Classes and index", train_data.class_to_idx)

    #check availability of gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    ## Build model 
    # Create CNN Model

    print('==> Building model..')
    model = SimpleDLA()

    # Write info to tensorboard
    summary(model, input_size=(3,32,32))
    writer=SummaryWriter('./content/logsdir')
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()

    model.to(device)
    # exit(0)

    if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

    if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('run/epochs/checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./run/epochs/checkpoint/ckpt.pth')
            model.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

    # test()
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    from utils import progress_bar
    def train(epoch):
        print('\nEpoch: %d / %d     lr = %.5f' % (epoch+1, EPOCH, scheduler.get_lr()[0]))
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        loss_his = train_loss/(batch_idx+1)
        acc_his = 100.*correct/total
        return loss_his, acc_his


    def test(epoch):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        loss_his = test_loss/(batch_idx+1)
        acc_his = 100.*correct/total

        # Save best checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('run/epochs/checkpoint'):
                os.mkdir('run/epochs/checkpoint')
            torch.save(state, './run/epochs/checkpoint/ckpt.pth')
            best_acc = acc

        return loss_his, acc_his


    def fit_model(num_epochs):
        # Traning the Model
        #history-like list for store loss & acc value
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []

        for epoch in range(start_epoch, start_epoch+num_epochs):
            train_loss, train_acc = train(epoch)
            test_loss, test_acc = test(epoch)
            scheduler.step()
            training_loss.append(train_loss)
            training_accuracy.append(train_acc)
            validation_loss.append(test_loss)
            validation_accuracy.append(test_acc)

        return training_loss, training_accuracy, validation_loss, validation_accuracy

    # Save directory
    if not os.path.exists("run/epochs"):
        os.makedirs("run/epochs")

    # Training
    training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(num_epochs=EPOCH)

    # # Save directory
    if not os.path.exists("run/epochs"):
        os.makedirs("run/epochs")

    # visualization
    plt.figure()
    plt.plot(range(EPOCH), training_loss, 'b-', label='Train')
    plt.plot(range(EPOCH), validation_loss, 'r-', label='Val')
    plt.title('Training & Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('run/epochs/History_loss.png')

    plt.figure()
    plt.plot(range(EPOCH), training_accuracy, 'b-', label='Train')
    plt.plot(range(EPOCH), validation_accuracy, 'r-', label='Val')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('run/epochs/History_acc.png')

    # plt.tight_layout()
    plt.show()
