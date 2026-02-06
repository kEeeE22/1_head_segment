import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,utils,models
import sys

class DFPmodel(torch.nn.Module):
    def __init__(self,pretrained=True,freeze=True):
        super(DFPmodel,self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initializeVGG(pretrained,freeze)

        ### Room Boundary Prediction
        rblist = [512,256,128,64,32,3]
        self.rbtrans = nn.ModuleList([self._transconv2d(
            rblist[i],rblist[i+1],4,2,1) for i in range(len(rblist)-2)])
        self.rbconvs = nn.ModuleList([self._conv2d(
            rblist[i],rblist[i+1],3,1,1) for i in range(len(rblist)-1)])
        self.rbgrs = nn.ModuleList([self._conv2d(
            rblist[i],rblist[i],3,1,1) for i in range(1,len(rblist)-1)])

        ### Room Type Prediction
        rtlist = [512,256,128,64,32]
        self.rttrans = nn.ModuleList([self._transconv2d(
            rtlist[i],rtlist[i+1],4,2,1) for i in range(len(rtlist)-1)])
        self.rtconvs = nn.ModuleList([self._conv2d(
            rtlist[i],rtlist[i+1],3,1,1) for i in range(len(rtlist)-1)])
        self.rtgrs = nn.ModuleList([self._conv2d(
            rtlist[i],rtlist[i],3,1,1) for i in range(1,len(rtlist))])

        # Attention Non-local context
        clist = [256,128,64,32]
        self.ac1s = nn.ModuleList(self._conv2d(
            clist[i],clist[i],3,1,1) for i in range(len(clist)))
        self.ac2s = nn.ModuleList(self._conv2d(
            clist[i],clist[i],3,1,1) for i in range(len(clist)))
        self.ac3s = nn.ModuleList(self._conv2d(
            clist[i],1,1,1) for i in range(len(clist)))
        self.xc1s = nn.ModuleList(self._conv2d(
            clist[i],clist[i],3,1,1) for i in range(len(clist)))
        self.xc2s = nn.ModuleList(self._conv2d(
            clist[i],1,1,1) for i in range(len(clist)))
        self.ecs = nn.ModuleList(self._conv2d(
            1,clist[i],1,1) for i in range(len(clist)))
        self.rcs = nn.ModuleList(self._conv2d(
            2*clist[i],clist[i],1,1) for i in range(len(clist)))
        
        # Direction aware kernel
        dak = [9,17,33,65]
        # horzontal
        self.hs = nn.ModuleList(self._dirawareLayer([1,1,dim,1]) 
                for dim in dak)
        # vertical
        self.vs = nn.ModuleList(self._dirawareLayer([1,1,1,dim])
                for dim in dak)
        # diagonal
        self.ds = nn.ModuleList(self._dirawareLayer([1,1,dim,dim],
            diag=True) for dim in dak)
        # diagonal flip
        self.dfs = nn.ModuleList(self._dirawareLayer([1,1,dim,dim],
            diag=True,flip=True) for dim in dak)
        # Last layer
        self.last = self._conv2d(clist[-1],9,1,1)
    
    def _dirawareLayer(self,shape,diag=False,flip=False,trainable=False):
        w = self.constant_kernel(shape,diag,flip,trainable)
        pad = ((np.array(shape[2:])-1)/2).astype(int)
        conv = nn.Conv2d(1,1,shape[2:],1,list(pad),bias=False)
        conv.weight = w
        return conv

    def _initializeVGG(self,pretrained,freeze):
        encmodel = models.vgg16(pretrained=pretrained)
        if freeze:
            for child in encmodel.children():
                for param in child.parameters():
                    param.requires_grad = False
        features = list(encmodel.features)[:31]
        self.features = nn.ModuleList(features)
    
    def _conv2d(self,in_,out,kernel,stride=1,padding=0):
        conv2d = nn.Conv2d(in_,out,kernel,stride,padding)
        nn.init.kaiming_uniform_(conv2d.weight)
        nn.init.zeros_(conv2d.bias)
        return conv2d

    def _transconv2d(self,in_,out,kernel,stride=1,padding=0):
        transconv2d = nn.ConvTranspose2d(in_,out,kernel,stride,padding)
        nn.init.kaiming_uniform_(transconv2d.weight)
        nn.init.zeros_(transconv2d.bias)
        return transconv2d

    def constant_kernel(self,shape,value=1,diag=False,
            flip=False,trainable=False):
        if not diag:
            k = nn.Parameter(torch.ones(shape)*value,requires_grad=trainable)
        else:
            w = torch.eye(shape[2],shape[3])
            if flip:
                w = torch.reshape(w,(1,shape[2],shape[3]))
                w = w.flip(0,1)
            w = torch.reshape(w,shape)
            k = nn.Parameter(w,requires_grad=trainable)
        return k

    def context_conv2d(self,t,dim=1,size=7,diag=False,
            flip=False,stride=1,trainable=False):
        N,C,H,W = t.size(0),t.size(1),t.size(2),t.size(3)
        in_dim = C
        size = size if isinstance(size,(tuple,list)) else [size,size]
        stride = stride if isinstance(stride,(tuple,list)) else [1,stride,stride,1]
        shape = [dim,in_dim,size[0],size[1]]
        w = self.constant_kernel(shape,diag=diag,flip=flip,trainable=trainable)
        pad = ((np.array(shape[2:])-1)/2).astype(int)
        conv = nn.Conv2d(1,1,shape[2:],1,list(pad),bias=False)
        conv.weight = w
        conv.to(self.device)
        return conv(t)

    def non_local_context(self,t1,t2,idx,stride=4):
        N,C,H,W = t1.size(0),t1.size(1),t1.size(2),t1.size(3)
        hs = H // stride if (H // stride) > 1 else (stride-1)
        vs = W // stride if (W // stride) > 1 else (stride-1)
        hs = hs if (hs%2!=0) else hs+1
        vs = hs if (vs%2!=0) else vs+1

        a = F.relu(self.ac1s[idx](t1))
        a = F.relu(self.ac2s[idx](a))
        a = torch.sigmoid(self.ac3s[idx](a))
        x = F.relu(self.xc1s[idx](t2))
        x = torch.sigmoid(self.xc2s[idx](x))
        x = a*x

        # direction-aware kernels
        h = self.hs[idx](x)
        v = self.vs[idx](x)
        d1 = self.ds[idx](x)
        d2 = self.dfs[idx](x)

        # double attention 
        c1 = a*(h+v+d1+d2)

        # expand channel
        c1 = self.ecs[idx](c1)

        # concatenation + upsample
        features = torch.cat((t2,c1),dim=1)

        out = F.relu(self.rcs[idx](features))
        return out

    def forward(self,x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {4,9,16,23,30}:
                results.append(x)
        rbfeatures = []
        for i,rbtran in enumerate(self.rbtrans):
            x = rbtran(x)+self.rbconvs[i](results[3-i])
            x = F.relu(self.rbgrs[i](x))
            rbfeatures.append(x)
        logits_cw = F.interpolate(self.rbconvs[-1](x),512)
        rtfeatures = []
        x = results[-1]
        for j,rttran in enumerate(self.rttrans):
            x = rttran(x)+self.rtconvs[j](results[3-j])
            x = F.relu(self.rtgrs[j](x))
            x = self.non_local_context(rbfeatures[j],x,j)
        
        logits_r = F.interpolate(self.last(x),512)

        return logits_r,logits_cw


def create_dfpmodel(pretrained=True, freeze=True, loadmodel=None, device='cpu'):
    model = DFPmodel(pretrained=pretrained, freeze=freeze):
    if loadmodel: 
        model.load_state_dict(torch.load(args.loadmodel))
    model.to(device)

    return model

def get_dfpcriterion():
    def criterion(logit_r, logit_b, r, b):
        w1, w2 = cross_two_tasks_weight(r, b)
        return w1 * balanced_entropy(logit_r, r) + w2 * balanced_entropy(logit_b, b)
    return criterion

if __name__ == "__main__":

    with torch.no_grad():
        testin = torch.randn(1,3,512,512)
        model = DFPmodel()
        model.load_state_dict(torch.load('weights650.pth'))
        ### Shared VGG encoder
        logits_r,logits_cw = model.forward(testin)
        # 0: 64x256x256, 1: 128x128x128
        # 2: 256x64x64, 3: 512x32x32, 4: 512x16x16
        print(logits_cw.size(),logits_r.size())
        breakpoint()
        gc.collect()


def balanced_entropy(preds,targets):
    eps = 1e-6
    m = nn.Softmax(dim=1)
    z = m(preds)
    cliped_z = torch.clamp(z,eps,1-eps)
    log_z = torch.log(cliped_z)
    num_classes = targets.size(1)
    ind = torch.argmax(targets,1).type(torch.int)

    total = torch.sum(targets)
    
    m_c,n_c = [],[]
    for c in range(num_classes):
        m_c.append((ind==c).type(torch.int))
        n_c.append(torch.sum(m_c[-1]).type(torch.float))

    c = []
    for i in range(num_classes):
        c.append(total-n_c[i])
    tc = sum(c)

    loss = 0
    for i in range(num_classes):
        w = c[i]/tc
        m_c_one_hot = F.one_hot((i*m_c[i]).permute(1,2,0).type(torch.long),
                num_classes)
        m_c_one_hot = m_c_one_hot.permute(2,3,0,1)
        y_c = m_c_one_hot*targets
        loss += w*torch.sum(-torch.sum(y_c*log_z,axis=2))
    return loss/num_classes

def cross_two_tasks_weight(rooms,boundaries):
    p1 = torch.sum(rooms).type(torch.float)
    p2 = torch.sum(boundaries).type(torch.float)
    w1 = torch.div(p2,p1+p2)
    w2 = torch.div(p1,p1+p2)
    return w1,w2