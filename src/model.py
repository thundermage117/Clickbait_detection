
import torch
from torch import nn,optim
import math 
import numpy as np
from torchmetrics.classification import BinaryAccuracy,BinaryF1Score,BinaryPrecision,BinaryRecall


def cos_sim(L_h,  L_b):
    # print(L_h.shape,L_b.shape)
#   print(torch.linalg.norm(L_h,dim=1).shape,torch.linalg.norm(L_b,dim=1).shape)
    cos_similarity=torch.nn.CosineSimilarity()
    ret=cos_similarity(L_h,L_b)
    # print("Cos_sim shape:",ret.shape)
    return ret
    # return torch.sum(L_h*L_b,dim=1)/(torch.linalg.norm(L_h,dim=1)*torch.linalg.norm(L_b,dim=1))

def R(H,B):
    r=cos_sim(H,B)
    # print("r:",r[0])
    dev = r.get_device()
    Rhb = torch.zeros((r.shape[0],2)).to(dev)
    # r1=r 
    # r2=1-r
    Rhb[:,0],Rhb[:,1]=r,1-r
    # print(Rhb.shape)
    softMax=torch.nn.Softmax(dim=1)   
    # return torch.exp(r1) / torch.sum(torch.exp(r1),dim=1),torch.exp(r2)/ torch.sum(torch.exp(r2),dim=1)
    # return softMax(Rhb)
    return Rhb

def loss_function(H,B,label):
    r=R(H,B)
    # print(r.shape)
    # print('label:',label[0])
    # print(r1.shape,r2.shape,label.shape)
    # print(r1[0],r2[0],label[0])
    # print("r_min",r.min())
    # Loss=-((1-label)*torch.log(r[:,0]+1e-15)+(label)*torch.log(r[:,1]+1e-15))
    # print("loss_in",Loss)
    
    # loss=(torch.sum(Loss))/Loss.shape[0]
    Loss_fn = nn.CrossEntropyLoss()
    loss=Loss_fn((1-r),label.long())
    return loss

def loss_function2(P,label):
    # print(r.shape)
    # print('label:',label[0])
    # print(r1.shape,r2.shape,label.shape)
    # print(r1[0],r2[0],label[0])
    # print('P_min:',(1-P).min(),'label_unique:',torch.unique(label),P.shape,label.shape)
    # P_lcl=P
    # P_lcl=(P-P.min())/(P.max()-P.min())
    # Loss=-((label)*torch.log(P[:,1]+1e-15)+(1-label)*torch.log(P[:,0]+1e-15))
    Loss_fn = nn.CrossEntropyLoss()
    # label1=torch.LongTensor(label)
    # print(f"P[0]:{P[:3]},Label[0]:{label[:3]}")

    loss=Loss_fn((P),label.long())
    # print(Loss.shape[0])
    # print(Loss.shape,label.shape,(1-P).shape)
    # loss=(torch.sum(Loss))/Loss.shape[0]
    # print(loss)
    return loss

class SimilarityAware(nn. Module):
    def __init__(self,in_size,hidden_size,out_size=None,n_layers=1):
        super().__init__()
        self.in_size=in_size
        self.hidden_size=hidden_size
        self.out_size=out_size
        if out_size is None:
            self.out_size=hidden_size
        self.n_layers=n_layers
        self.bi_gru=nn.GRU(self.in_size,self.hidden_size,batch_first=True,num_layers=self.n_layers,bidirectional=True)
        self.att=nn.Linear(2*(self.hidden_size),1)
        self.tanh=nn.Tanh()
        self.softMax=nn.Softmax(dim=1)

    def forward(self,x,h=None):
        if h is not None:
            out,h=self.bi_gru(x,h)
        else:
            out,h=self.bi_gru(x)
        # out=torch.cat(out[:,:,:self.hidden_size],out[:,:,self.hidden_size:])
        # print('x_min',x.min())
        # print("out_min:",out.min())
        u=self.att(out)
        ut=self.tanh(u)
        a=self.softMax(ut)
        # print(ut.shape,a.shape,out.shape)
        # print(out.shape)
        out=torch.sum(out*a,dim=1)
        # print(a.shape,ut.shape,out.shape)
        return out

class myLinear(nn.Module):
    def __init__(self, in_features, out_features, bias_in=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias_in
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias_in:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias=None
            # self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        y = input.shape[-1]
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        # print('my_layer:',self.weight.shape,(input).shape)
        output = torch.matmul(input.mT,self.weight).mT
        # print("myLayer:",output.shape)
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
class LSD(nn.Module):
    def __init__(self,in_size,hidden_size,out_size=None,n_layers=1):
        super().__init__()
        self.head_Model=SimilarityAware(in_size,hidden_size,out_size,n_layers)
        self.body_Model=SimilarityAware(in_size,hidden_size,out_size,n_layers)
        
    def forward(self,head,body):
        Lh=self.head_Model(head)
        Lb=self.body_Model(body)
        return Lh,Lb

class LSDA(nn.Module):
    def __init__(self,in_size,hidden_size,K_size,d_a=None,out_size=None,n_layers=1,g_and_l=True):
        super().__init__()
        self.LSD_model=LSD(in_size,hidden_size,out_size=out_size,n_layers=n_layers)
        self.mlpLayer=nn.Linear(2*(hidden_size),K_size)
        att_in_size=3*K_size if g_and_l is True else K_size
        d_a=d_a if d_a is not None else att_in_size
        r=d_a
        # self.att1=nn.Linear(att_in_size,d_a,bias=False)
        # self.att1=myLinear(in_features=att_in_size,out_features=d_a,bias=False)
        self.att1=myLinear(att_in_size,1,bias_in=False)
        # self.att2=nn.Linear(d_a,r,bias=False)
        # self.att2=nn.Linear(d_a**2,r,bias=False)
        self.att2=myLinear(att_in_size,d_a,bias_in=False)

        self.mlpLayer2=nn.Linear(d_a,2)
        # self.att2=nn.Linear()
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        # self.softMax=nn.Softmax(dim=2)
        self.softMax1=nn.Softmax(dim=1)
        # self.softMax2=nn.Softmax(dim=0)
        self.K_size=K_size
        self.g_and_l=g_and_l
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,head,body):
        Lh,Lb=self.LSD_model(head,body)
        # print('Lh_min:',Lh.min())
        Lh_blocks=torch.tensor_split(Lh,self.K_size,dim=-1)
        Lb_blocks=torch.tensor_split(Lb,self.K_size,dim=-1)
        Lsb=torch.zeros((Lh.shape[0],self.K_size)).to(self.device)
        for idx,Lh_block in enumerate(Lh_blocks):
            Lb_block=Lb_blocks[idx]
            # print(Lsb.shape,'Lb_block:',Lb_block.shape,cos_sim(Lh_block,Lb_block).shape)
            Lsb[:,idx]=cos_sim(Lh_block,Lb_block)
            # if idx==0:
                # print(cos_sim(Lh_block,Lb_block).shape)
        if self.g_and_l is True:
            Lhk=self.mlpLayer(Lh)
            Lbk=self.mlpLayer(Lb)
            L_comb=torch.zeros((Lb.shape[0],3*self.K_size)).to(self.device)
            # print(L_comb[:,:self.K_size].shape,Lhk.shape)
            L_comb[:,:self.K_size]=Lhk
            L_comb[:,self.K_size:2*self.K_size]=Lsb
            L_comb[:,2*self.K_size:]=Lbk
        else:
            L_comb=Lsb
        transformed_comb=L_comb.reshape(L_comb.shape[0],1,-1)
        hbar=self.tanh(self.att1(transformed_comb))

        # hbar=hbar.reshape(hbar.shape[0],-1)
        hbar2=self.att2(hbar)

        # print('hbar_shape:',hbar.shape,'hbar2_shape:',hbar2.shape)
        sz=hbar2.shape
        A_tmp=self.softMax1(hbar2.reshape(hbar2.shape[0],-1))
        # A_tmp=torch.exp(hbar2-hbar2.max(axis=torch.tensor((1,2))))/torch.exp(hbar2-hbar2.max(axis=torch.tensor((1,2)))).sum(axis=torch.tensor((1,2)))
        A_tmp=A_tmp.reshape(sz)
        # print("Softmax_sum:",A_tmp[0].sum())
        A_prod=torch.bmm(A_tmp,transformed_comb.mT).mT
        # print("A_prod:",A_prod.shape,"Transform_comb:",transformed_comb.shape)
        A_prod=A_prod.reshape(A_prod.shape[0],-1)

        A_mlp=self.mlpLayer2(A_prod)
        # A_sig=self.sigmoid(A_mlp)
        # P=self.softMax2(A_sig)
        # P=self.softMax1(A_mlp)
        P=A_mlp
        # print("P_grad_fn:",P.grad_fn)
        return P,Lh,Lb


def train(model,head,body,labels,batch_size=10,n_epochs=10,lr=0.1,lm=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert(head.shape[0]==body.shape[0] and body.shape[0]==labels.shape[0])
    adam=optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(adam, step_size=n_epochs//10, gamma=0.5)
    for epoch in range(n_epochs):
        idxs=torch.randint(low=0,high=body.shape[0],size=(batch_size,))
        head_lcl=head[idxs,...]
        body_lcl=body[idxs,...]
        label_lcl=labels[idxs]
        adam.zero_grad()
        Lh,Lb=model(head_lcl,body_lcl)
        # print(f"Lh={Lh.shape}::Lb={Lb.shape}")
        loss=loss_function(Lh,Lb,label_lcl)
        print(f"Epoch:{epoch}::Loss={loss}")
        loss.backward()
        adam.step()
        # scheduler.step()

def train2(model,head,body,labels,batch_size=10,n_epochs=10,lr=0.1,lm=1e-3,gamma=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert(head.shape[0]==body.shape[0] and body.shape[0]==labels.shape[0])

    adam=optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(adam, step_size=n_epochs//10, gamma=gamma)
    losses=np.zeros(n_epochs)
    accuracies=np.zeros(n_epochs)
    f1_score=np.zeros(n_epochs)
    recall=np.zeros(n_epochs)
    precision=np.zeros(n_epochs)
    model=model.train()
    binary_acc=BinaryAccuracy().to(device)
    binary_f1=BinaryF1Score().to(device)
    binary_recall=BinaryRecall().to(device)
    binary_precision=BinaryPrecision().to(device)
    for epoch in range(n_epochs):
        idxs=torch.randint(low=0,high=body.shape[0],size=(batch_size,))
        head_lcl=head[idxs,...]
        body_lcl=body[idxs,...]
        label_lcl=labels[idxs]
        adam.zero_grad()
        P,Lh,Lb=model(head_lcl,body_lcl)
        # print("P-shape:",P.shape)
        # print(f"Lh={Lh.shape}::Lb={Lb.shape}")
        # print("Lh,Lb:",Lh.shape,Lb.shape)
        lossG=loss_function(Lh,Lb,label_lcl)
        lossL=loss_function2(P,label_lcl)
        score_acc=binary_acc(P.argmax(dim=1),label_lcl)
        f1_score[epoch]=binary_f1(P.argmax(dim=1),label_lcl)
        recall[epoch]=binary_recall(P.argmax(dim=1),label_lcl)
        precision[epoch]=binary_precision(P.argmax(dim=1),label_lcl)
        loss=lossL+lossG
        accuracies[epoch]=score_acc
        losses[epoch]=loss
        print(f"Epoch:{epoch}::Loss={loss},lossL:{lossL}")
        # print("Requires grad",lossL.require_grad())
        loss.backward()
        adam.step()
        scheduler.step()
    return losses,accuracies,f1_score,recall,precision