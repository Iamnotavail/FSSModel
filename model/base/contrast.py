import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Contrast_Activate(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, node=(5, 2)):
        "Take in model size and number of heads."
        super(Contrast_Activate, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.linear_q1 = nn.Linear(d_model, d_model)
        self.linear_q2 = nn.Linear(d_model, d_model)
        # self.linear_foreground = nn.Linear(1, 1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.tree_drop=nn.Dropout(p=0.3)
        self.node_num=node
        self.tree1=PMMs(c=d_model,k=self.node_num[0])
        self.tree2=PMMs(c=d_model,k=self.node_num[1])
        #self.tree3=PMMs(c=d_model,k=self.node_num[2])
        self.node_w1=NodeWeight()
        self.node_w2=NodeWeight()
        #self.node_w3 = NodeWeight()
        self.Conv1D=Conv_1D()
        self.relu=nn.ReLU()

    def forward(self, query, key, value, lamda, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query_l, key_l = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]
        # value=(bsz,h*w)
        # ------mask权重------
        # weight=torch.sum(value,dim=-1,keepdim=True)/(value.size()[-1])
        # value=value/(weight.expand_as(value)+1e-6)
        # -------------------
        value_l = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
        # value=(bsz,head,h*w,1)
        # ---------加上背景----------------
        #value = torch.cat([value, torch.ones_like(value)-value],dim=-1)
        # -------------------------------

        # --------整体信息--------
        key_glbal=torch.mean(key_l*value_l.expand_as(key_l),dim=-2,keepdim=True)
        # scale = torch.sum(value, dim=-2, keepdim=True)/value.size()[-2]
        # key_glbal=key_glbal/scale
        # key_glbal=(bsz,head,1,c)
        d_k = query_l.size(-1)
        whole = query_l*key_glbal.expand_as(query_l)
        whole = torch.sum(whole,dim=-1,keepdim=True)/ math.sqrt(d_k)
        # whole=(bsz,head,hw,1)
        whole = self.relu(whole)
        whole = torch.mean(whole,-3)

        part, self.attn = attention(query_l, key_l, value_l, mask=mask,
                                 dropout=self.dropout)
        part=torch.mean(part, -3)

        f=(1-lamda)*part+lamda*whole
        #f=self.Conv1D(torch.cat((part,whole),dim=-1))
        #f=part

        # bsz,hw,ch=query.size()
        # w=math.sqrt(hw)
        # w=int(w)
        # q=query.permute(0, 2, 1).contiguous().view(bsz, ch, w, w)

        # -----parsing tree---------
        q = query.permute(0, 2, 1).contiguous()
        q1,lq1=self.tree1(q,query)  # l1(b,c,node_n) query(b,n,c) --> q1(b,n,node_n)
        q2,lq2 = self.tree2(lq1,query)
        #q3,lq3=self.tree3(lq2,query)

        # k=key.permute(0, 2, 1).contiguous()
        # k1,lk1=self.tree1(k,key)
        # k2,lk2=self.tree2(lk1,key)
        #
        # wq1=cal_weight(lq1,lk1,k1,value)
        # wq2=cal_weight(lq2,lk2,k2,value)
        # q_out=torch.cat((q1*wq1.expand_as(q1),q2*wq2.expand_as(q2)),dim=-1)
        q1 = self.node_w1(q1)
        q2 = self.node_w2(q2)
        #q3 = self.node_w3(q3)
        q_out = torch.cat((q1, q2), dim=-1)
        q_out=self.tree_drop(q_out)
        #q_out=q1

        return torch.cat((f,q_out),dim=-1)

class Conv_1D(nn.Module):
    def __init__(self):
        super(Conv_1D, self).__init__()
        self.Conv=nn.Conv1d(in_channels=2,out_channels=1,kernel_size=1)

    def forward(self,x):
        x=x.permute(0, 2, 1).contiguous()
        x=self.Conv(x)
        x=x.permute(0, 2, 1).contiguous()
        return x


class NodeWeight(nn.Module):
    def __init__(self, mid_ch=16):
        super(NodeWeight, self).__init__()
        self.Conv=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=mid_ch,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_ch, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
            )
        # self.Avg=nn.AdaptiveAvgPool2d(1)
        # self.Max=nn.AdaptiveMaxPool2d(1)
        # self.Sigm=nn.Sigmoid()

    def forward(self,x):
        bsz,hw,ch=x.size()
        w=math.sqrt(hw)
        w=int(w)

        y=x.permute(0, 2, 1).contiguous().view(bsz*ch,1, w, w)
        y=self.Conv(y)
        # y=self.Sigm(y)
        y=y.view(bsz,ch,-1).permute(0, 2, 1).contiguous()

        return x*y

def cal_weight(lq,lk,k,v):
    '''
    input: lq(b,c,node_n) lk(b,c,node_n) k(b,n,node_n) v(b,n)
    output: wq(b,1,node_n)
    '''
    with torch.no_grad():
        wk=k*v.unsqueeze(-1).expand_as(k)  #(b,n,node_n)
        wk=torch.mean(wk,dim=1,keepdim=True)  #(b,1,node_n)
        score=torch.bmm(lq.transpose(-2,-1),lk)
        score=F.softmax(score, dim=-1)  #(b,node_n,node_n)
        wq=torch.matmul(score,wk.transpose(-2,-1))
        #wq=torch.max(qi*p.expand_as(qi),dim=1,keepdim=True)[0].data
    return wq.transpose(-2,-1)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #q= torch.Size([8, 8, 2304, 32]) k= torch.Size([8, 8, 2304, 32]) v= torch.Size([8, 8, 2304, 1])
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # --------------------
    p_attn = F.softmax(scores, dim=-1)
    #q= torch.Size([8, 8, 2304, 2304])
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PMMs(nn.Module):
    '''Prototype Mixture Models
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k=10, stage_num=10):
        super(PMMs, self).__init__()
        self.stage_num = stage_num
        self.num_pro = k
        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init mu
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        self.kappa = 20

    def forward(self,node, query_feature):
        '''
        input: node(b,c,n) query_feature(b,n,c)
        output: Prob_map(b,n,k) mu1(b,c,k)
        '''
        # mu1 = self.generate_prototype(node)
        mu1=self.EM(node)  # node(b,c,n) --> mu1(b,c,k)
        Prob_map = self.discriminative_model(query_feature, mu1)  # query_f(b,n,c) --> Prob_map(b,n,k)

        return Prob_map,mu1

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def EM(self,x):
        '''
        EM method
        :param x: feauture  b * c * n
        :return: mu
        '''
        b = x.shape[0]
        # mu = self.mu.normal_(0, math.sqrt(2. / self.num_pro))  # Randomly init mu
        # mu = self._l2norm(mu, dim=1)
        # print(torch.max(mu).item())
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                # E STEP:
                z = self.Kernel(x, mu)
                z = F.softmax(z, dim=2)  # b * n * k
                # M STEP:
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k

                mu = self._l2norm(mu, dim=1)

        #mu = mu.permute(0, 2, 1)  # b * k * c

        return mu  # b * c * k

    def Kernel(self, x, mu):
        x_t = x.permute(0, 2, 1)  # b * n * c
        z = self.kappa * torch.bmm(x_t, mu)  # b * n * k

        return z

    def discriminative_model(self, x, mu):  #, mu_f, mu_b):
        '''
        input: x(b,n,c), mu(b,c,k)
        output: P(b,n,k)
        '''

        #mu = torch.cat([mu_f, mu_b], dim=1)
        # mu = mu.permute(0, 2, 1)

        #b, c, h, w = query_feature.size()
        #x = query_feature.view(b, c, h * w)  # b * c * n
        with torch.no_grad():

            #x_t = x.permute(0, 2, 1)  # b * n * c
            z = torch.bmm(x, mu)  # b * n * k

            P = F.softmax(z, dim=2)  # b * n * k

        #P = z.permute(0, 2, 1)

        #P = P.view(b, self.num_pro, h, w) #  b * k * w * h  probability map
        # P_f = torch.sum(P[:, 0:self.num_pro], dim=1).unsqueeze(dim=1) # foreground
        # P_b = torch.sum(P[:, self.num_pro:], dim=1).unsqueeze(dim=1) # background
        #
        # Prob_map = torch.cat([P_b, P_f], dim=1)

        return P  #Prob_map, P

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
