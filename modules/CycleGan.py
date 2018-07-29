import torch
import torch.nn as nn
import numpy as np
import os
import itertools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        #print(x.shape)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ResConvBlock(nn.Module):
    def __init__(self, depth,use_bias=False):
        super(ResConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(depth, depth, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(depth),
            nn.ReLU(True),
            nn.Conv2d(depth, depth, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(depth)
        )

    def forward(self, x):
        out=x+self.model(x)
        return out

class Generator(nn.Module):
    def __init__(self,img_height, img_width, img_channel,depth=32,learning_rate = 2e-4):
        super(Generator, self).__init__()
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False

        self.model = nn.Sequential(
            # down sampling
            nn.ReflectionPad2d(3),
            Print(),
            #c7s1-k
            nn.Conv2d(img_channel, depth * 1, 7, stride=1,padding=0),
            nn.InstanceNorm2d(depth * 1),
            nn.ReLU(inplace=True),
            Print(),
            
            #dk-64
            nn.Conv2d(depth * 1, depth * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(depth * 2),
            nn.ReLU(inplace=True),
            Print(),
            
            #dk-128
            nn.Conv2d(depth * 2, depth * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(depth * 4),
            nn.ReLU(inplace=True),
            Print(),
            # residual block
            ResConvBlock(depth=depth * 4),
            nn.ReLU(inplace=True),
            ResConvBlock(depth=depth * 4),
            nn.ReLU(inplace=True),
            ResConvBlock(depth=depth * 4),
            nn.ReLU(inplace=True),
            ResConvBlock(depth=depth * 4),
            nn.ReLU(inplace=True),
            ResConvBlock(depth=depth * 4),
            nn.ReLU(inplace=True),
            ResConvBlock(depth=depth * 4),
            nn.ReLU(inplace=True),
            Print(),
            # upsampling block
            #uk-64
            nn.ConvTranspose2d(depth * 4, depth * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(depth * 2),
            nn.ReLU(inplace=True),
            Print(),
            #uk-32
            nn.ConvTranspose2d(depth * 2, depth * 1, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(depth * 1),
            nn.ReLU(inplace=True),
            Print(),
            #nn.ReflectionPad2d(3),
            #Print(),
            #c7s1-k
            nn.ConvTranspose2d(depth * 1,img_channel, 7, stride=1, padding=3),
            nn.InstanceNorm2d(img_channel),
            nn.ReLU(inplace=True),
            Print(),
            #FunctionalPad(pad=(-3,-3,-3,-3)),
            #Print()
            
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        

    def forward(self, x):
        #print('generater')
        output = self.model(x)
        return output

    def step(self):
        self.optimizer.step()

    def clear_grad(self):
        self.optimizer.zero_grad()
    
    def setLearningRate(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class Discriminator(nn.Module):
    def __init__(self,img_height, img_width, img_channel,depth=64,learning_rate = 2e-4):
        super(Discriminator, self).__init__()

        #asuming its a square image, calculating last layer ksize
        lastKSize=img_height//(2*2*2*2)
        self.model = nn.Sequential(
            # down sampling
            nn.Conv2d(img_channel, depth * 1, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            Print(),
            nn.Conv2d(depth * 1, depth * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(depth * 2),
            nn.LeakyReLU(0.2, True),
            Print(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(depth * 4),
            nn.LeakyReLU(0.2, True),
            Print(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(depth * 8),
            nn.LeakyReLU(0.2, True),
            Print(),
            nn.Conv2d(depth * 8, depth * 8, lastKSize, stride=1, padding=0),
            nn.Tanh(),
            Print(),
            Flatten(),
            Print(),
            nn.Linear(depth * 8,1),
            nn.Sigmoid()
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self,x):
        output = self.model(x)
        return output

    def step(self):
        self.optimizer.step()

    def clear_grad(self):
        self.optimizer.zero_grad()
        
    def setLearningRate(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        


class CycleGan(object):
    def __init__(self, img_height, img_width, img_channel,lamda=100,batchSize=10,bufferSize=10, lr=1e-4,ganLossCoff=0.8,cycleLossCoff=2):
        self.FG = None
        self.BG = None
        self.D = None
        self.BD = None
        
        self.lr=lr
        self.lamda=lamda
        
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        
        self.criterionGAN = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        #self.criterionCycle = torch.nn.MSELoss()
        self.criterionIdt = torch.nn.L1Loss()
        #self.criterionIdt = torch.nn.MSELoss()
        
        self.cycleLossCoff=cycleLossCoff
        
        self.batchSize=batchSize
        self.bufferSize=bufferSize
        self.genBuffer=np.zeros([bufferSize,img_channel,img_height,img_width],dtype=np.float)
        self.bGenBuffer=np.zeros([bufferSize,img_channel,img_height,img_width],dtype=np.float)
        

    def _createGenerater(self):
        return Generator(self.img_height, self.img_width, self.img_channel,learning_rate = self.lr)

    def _createDiscriminator(self):
        return Discriminator(self.img_height, self.img_width, self.img_channel, learning_rate = self.lr)

    def generaterModel(self):
        if self.FG is not None:
            return self.FG
        self.FG = self._createGenerater()
        if torch.cuda.is_available():
            self.FG=self.FG.cuda()
        return self.FG

    def discriminator(self):
        if self.D is not None:
            return self.D
        self.D = self._createDiscriminator()
        if torch.cuda.is_available():
            self.D=self.D.cuda()
        return self.D

    def backwardGeneraterModel(self):
        if self.BG is not None:
            return self.BG
        self.BG = self._createGenerater()

        if torch.cuda.is_available():
            self.BG=self.BG.cuda()
        return self.BG
    
    def backwardDiscriminator(self):
        if self.BD is not None:
            return self.BD
        self.BD = self._createDiscriminator()
        if torch.cuda.is_available():
            self.BD=self.BD.cuda()
        return self.BD
    
    def setLearningRate(self, lr):
        
        gen=self.generaterModel()
        bGen=self.backwardGeneraterModel()
        dis=self.discriminator()
        bDis=self.backwardDiscriminator()
        
        gen.setLearningRate(lr)
        bGen.setLearningRate(lr)
        dis.setLearningRate(lr)
        bDis.setLearningRate(lr)
        
    
    def setInputs(self,x,y):
        self.realX=x
        self.realY=y
        
    def forward(self):
        
        self.recX=bGen(gen(self.realX))
        self.recY=gen(bGen(self.realY))
        return self.recX,self.recY
        
    
    def backward(self,datasetX,datasetY):
        
        loss_gen=0
        loss_cycleFW=0
        loss_disFw=0
        
        
        datasetX=datasetX.astype(np.float)/255
        datasetY=datasetY.astype(np.float)/255
        
        if torch.cuda.is_available():
            X=torch.from_numpy(datasetX).type('torch.cuda.FloatTensor')
            Y=torch.from_numpy(datasetY).type('torch.cuda.FloatTensor')
        else:
            X=torch.from_numpy(datasetX).type('torch.FloatTensor')
            Y=torch.from_numpy(datasetY).type('torch.FloatTensor')
        
        X=X.permute(0,3,1,2)
        Y=Y.permute(0,3,1,2)
        
        gen=self.generaterModel()
        bGen=self.backwardGeneraterModel()
        dis=self.discriminator()
        bDis=self.backwardDiscriminator()
        
        gen.clear_grad()
        bGen.clear_grad()
        dis.clear_grad()
        bDis.clear_grad()
        
        y1 = np.ones([datasetX.shape[ 0 ], 1 ])
        y0 = np.zeros([self.bufferSize, 1 ])
        #y0 = np.zeros([datasetX.shape[ 0 ], 1 ])
        if torch.cuda.is_available():
            y1 = torch.from_numpy(y1).type('torch.cuda.FloatTensor')
            y0 = torch.from_numpy(y0).type('torch.cuda.FloatTensor')
        else:
            y1 = torch.from_numpy(y1).type('torch.FloatTensor')
            y0 = torch.from_numpy(y0).type('torch.FloatTensor')

        
        
        # forward generater
        self.set_requires_grad([dis,bDis],False)
        
        loss_gen=self.criterionGAN(dis(gen(X.detach())),y1.detach())
        #loss_gen*=self.ganLossCoff
        #loss_gen.backward()
        
        loss_bGen=self.criterionGAN(bDis(bGen(Y.detach())),y1.detach())
        #loss_bGen*=self.ganLossCoff
        #loss_bGen.backward()
        
        
        
        
        #cycle loss
        
        loss_cycleFW=self.criterionCycle(bGen(gen(X.detach())),X.detach())*self.lamda
        #loss_cycleFW.backward()
        
        loss_cycleBW=self.criterionCycle(gen(bGen(Y.detach())),Y.detach())*self.lamda
        #loss_cycleBW.backward()
        
        #identity Loss
        #loss_idtFW=self.criterionCycle(gen(X.detach()),Y.detach())*self.lamda*0.1
        #loss_idtFW.backward()
        
        #loss_idtBW=self.criterionCycle(bGen(Y.detach()),X.detach())*self.lamda*0.1
        #loss_idtBW.backward()
        
        lossGenerative=loss_gen+loss_bGen+loss_cycleFW+loss_cycleBW
        lossGenerative.backward()
        
        # discriminator train
        self.set_requires_grad([dis,bDis],True)
        GeneratedY=gen(X).cpu().detach().numpy()[np.random.randint(0,self.batchSize)]
        GeneratedX=bGen(Y).cpu().detach().numpy()[np.random.randint(0,self.batchSize)]
        
        self.genBuffer=np.roll(self.genBuffer,1,0)
        self.bGenBuffer=np.roll(self.bGenBuffer,1,0)
        
        self.genBuffer[0]=GeneratedY.copy()
        self.bGenBuffer[0]=GeneratedX.copy()
        
        if torch.cuda.is_available():
            genBuffer = torch.from_numpy(self.genBuffer.copy()).type('torch.cuda.FloatTensor')
            bGenBuffer = torch.from_numpy(self.bGenBuffer.copy()).type('torch.cuda.FloatTensor')
        else:
            genBuffer = torch.from_numpy(self.genBuffer.copy()).type('torch.FloatTensor')
            bGenBuffer = torch.from_numpy(self.bGenBuffer.copy()).type('torch.FloatTensor')

        
        
        loss_disFwReal=self.criterionGAN(dis(Y.detach()),y1.detach())
        loss_disFwFake=self.criterionGAN(dis(genBuffer),y0.detach())*((1.0*self.batchSize)/self.bufferSize)
        #loss_disFwFake=self.criterionGAN(dis(gen(X.detach()).detach()),y0.detach())
        loss_disFw=(loss_disFwReal+loss_disFwFake)*0.7
        loss_disFw.backward()
        

        
        loss_disBwReal=self.criterionGAN(bDis(X.detach()),y1.detach())
        loss_disBwFake=self.criterionGAN(bDis(bGenBuffer),y0.detach())*((1.0*self.batchSize)/self.bufferSize)
        #loss_disBwFake=self.criterionGAN(bDis(bGen(Y.detach()).detach()),y0.detach())
        loss_disBw=(loss_disBwReal+loss_disBwFake)*0.7
        loss_disBw.backward()
        

        gen.step()
        bGen.step()
        dis.step()
        bDis.step()
        
        return loss_gen,loss_cycleFW,loss_disFw

    def train_on_batch(self, datasetX, datasetY):
        return self.backward(datasetX,datasetY)

    def generate(self,inp):
        if torch.cuda.is_available():
            X=torch.from_numpy(inp).type('torch.cuda.FloatTensor')
        else:
            X=torch.from_numpy(inp).type('torch.FloatTensor')
        X=X.permute(0,3,1,2)
        gen=self.generaterModel()
        generated=gen(X)
        generated=generated.permute(0,2,3,1)
        
        bGen=self.backwardGeneraterModel()
        
        recon=bGen(gen(X).detach())
        recon=recon.permute(0,2,3,1)
        
        return generated.cpu().detach().numpy()*255, recon.cpu().detach().numpy()*255
    
    
    def bGenerate(self,inp):
        if torch.cuda.is_available():
            X=torch.from_numpy(inp).type('torch.cuda.FloatTensor')
        else:
            X=torch.from_numpy(inp).type('torch.FloatTensor')
        X=X.permute(0,3,1,2)
        gen=self.generaterModel()
        bGen=self.backwardGeneraterModel()
        generated=bGen(X)
        generated=generated.permute(0,2,3,1)
        
        
        recon=gen(bGen(X).detach())
        recon=recon.permute(0,2,3,1)
        
        return generated.cpu().detach().numpy()*255, recon.cpu().detach().numpy()*255
    
    def loadModel(self,PATH):
        if(os.path.exists(path=PATH+"/FG.mdl")):
            self.FG=torch.load(os.path.join(PATH,"FG.mdl"))
            self.D=torch.load(os.path.join(PATH,"D.mdl"))
            self.BG=torch.load(os.path.join(PATH,"BG.mdl"))
            self.BD=torch.load(os.path.join(PATH,"BD.mdl"))
            
            
    def saveModel(self,PATH):
        if not os.path.exists(path=PATH):
            os.mkdir(PATH)
        torch.save(self.FG, os.path.join(PATH,"FG.mdl"))
        torch.save(self.D, os.path.join(PATH,"D.mdl"))
        torch.save(self.BG, os.path.join(PATH,"BG.mdl"))
        torch.save(self.BD, os.path.join(PATH,"BD.mdl"))
        
        
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
