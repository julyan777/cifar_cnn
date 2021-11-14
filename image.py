import torch
import torchvision
import torch.nn as nn
import datetime
import torch.optim as optim
import argparse
import torchvision.transforms as transforms


loss_func = nn.CrossEntropyLoss()
best_acc = 0
ROOT = './data'
WORKERS = 4


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def net_train(net, train_data_loader, optimizer):
    net.train()
    begin = datetime.datetime.now()

    for i, data in enumerate(train_data_loader, 0):
        image, label = data
        image, label = image.cuda(), label.cuda()


        optimizer.zero_grad()

        outs = net(image)
        loss = loss_func(outs, label)
        loss.backward()
        optimizer.step()

    end = datetime.datetime.now()
    print('one epoch spend: ', end - begin)


def net_test(net, test_data_loader, epoch):
    net.eval()

    ok = 0

    for i, data in enumerate(test_data_loader):
        image, label = data
        image, label = image.cuda(), label.cuda()

        outs = net(image)
        _, pre = torch.max(outs.data, 1)
        ok += (pre == label).sum()

    acc = ok.item() * 100. / (len(test_data_loader.dataset))

    print('Epoch: {}, Acc:{}\n'.format(epoch, acc))
    global best_acc
    if acc > best_acc:
        best_acc = acc


def main():
    # 训练超参数设置，可通过命令行设置
    parser = argparse.ArgumentParser(description='PyTorch CIFA10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--epochs', type=int, default=2, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    train_data = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=ROOT, train=False, download=False, transform=transform)

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=WORKERS)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False,
                                            num_workers=WORKERS)

    net = LeNet().cuda()


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        net_train(net, train_load, optimizer)

        # 每个epoch结束后用测试集检查识别准确度
        net_test(net, test_load, epoch)

    end_time = datetime.datetime.now()

    global best_acc
    print('CIFAR10 pytorch LeNet Train: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(args.epochs, args.batch_size,
                                                                                     args.lr, best_acc))
    print('train spend time: ', end_time - start_time)


if __name__ == '__main__':
    main()
