import torch.nn as nn


class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)


def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


class Lenet(nn.Module):
    name = 'lenet'

    def __init__(self, opt, c1=20, c2=50, c3=100):
        super().__init__()

        opt['d'] = opt.get('d', 0.5)
        opt['l2'] = opt.get('l2', 0.0)

        bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d
        output_dim = 1

        def convbn(ci,co,ksz,psz):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz, padding=(1, 2), stride=(2, 4)),
                #bn2(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz, padding=1))
                #nn.Dropout(opt['d']))

        self.m = nn.Sequential(
            convbn(1,c1,5,3),
            convbn(c1,c2,5,2),
            View(c2*8*14),
            nn.Linear(c2*8*14, c3),
            # #bn1(c3),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c3,output_dim),
            nn.Sigmoid())

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)

    def forward(self, x):
            return self.m(x)


class Allcnn(nn.Module):
    name = 'allcnn'

    def __init__(self, opt, c = 1, c1=48, c2=96): #c1=96, c2=192, microbn=False):
        super().__init__()

        if (not 'd' in opt) or opt['d'] < 0:
            if opt['augment']:
                opt['d'] = 0.0
            else:
                opt['d'] = 0.5

        output_dim = 1

        bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d

        def convbn(ci,co,ksz,s,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                #bn2(co),
                nn.ReLU(True)
                )
        self.m = nn.Sequential(
            convbn(c,c1,3,2,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            convbn(c2,c2,3,1,1),
            nn.Dropout(opt['d']),
            convbn(c2,1,3,2,1),
            #convbn(c2,output_dim,3,2,1),
            #convbn(c2,output_dim,1,1))
            #nn.AvgPool2d((6, 40)),
            View(11*80),
            nn.Linear(11*80, output_dim),
            nn.Sigmoid())

    def forward(self, x):
        return self.m(x)