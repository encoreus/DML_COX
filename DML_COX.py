import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot  as plt
import matplotlib
import copy

from random import shuffle
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset,ConcatDataset
# from torch.distributions.multivariate_normal import MultivariateNormal,Normal
from torch.distributions import MultivariateNormal,Normal
from torch import nn
from torchvision import datasets,transforms
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(device)


# standardize the data
def feature_normalize(data):
    mu = torch.mean(data,axis=0)
    std = torch.std(data,axis=0)
    return (data - mu)/std

'''
Define the survival data class:
    T: failure time, the smaller value of the death time and censoring time
    D: censoring indicator, 0: censored, 1: uncensored
    X: linear covariates of interest
    Z: confounding covariates
'''
class DeepSurvData:
    def __init__(self,T,D,X,Z,device):
        self.T =T.to(device)
        self.D =D.to(device)
        self.X =X.to(device)
        self.Z =Z.to(device)
        self.device = device
        pass
    
    def cpu(self):
        device = 'cpu'
        self.T = self.T.to(device)
        self.D = self.D.to(device)
        self.X = self.X.to(device)
        self.Z = self.Z.to(device)
        self.device = device
        return self
    
    def to(self,device):
        self.T = self.T.to(device)
        self.D = self.D.to(device)
        self.X = self.X.to(device)
        self.Z = self.Z.to(device)
        self.device = device
        return self

    def __getitem__(self,subset):
        device= self.device
        T=self.T[subset,:]
        D=self.D[subset,:]
        X=self.X[subset,:]
        Z=self.Z[subset,:]
        return DeepSurvData(T,D,X,Z,device)
        
    pass

# Define the class to record the information of nuisance estimator theta(Z) 
# and Nelson–Aalen estimator Lambda(t) for each observation
class DeepSurv:
    def __init__(self, sim_data, beta, theta, device):
        sim_data = sim_data.to(device)
        beta=beta.to(device)
        
        T = sim_data.T
        D = sim_data.D
        X = sim_data.X
        Z = sim_data.Z

        T_mat = (T<=torch.squeeze(T,1)).to(torch.float32)
        
        # self.T = T[D==1]
        self.T = T
        self.D = D
        self.X = X
        self.Z = Z
        self.T_D1 = T[D==1]
        
        self.theta = theta
        self.hratio = 1/(T_mat @ torch.exp(theta(Z).to(device)+ X @ torch.unsqueeze(beta,1)))[D==1]
        self.device = device
        pass

    def plot(self, theta, Baseline = False, Z_new=False, save_fig=False):
        device=self.device
        theta_hat = self.theta
        Z_dim=self.Z.shape[1]
        
        if isinstance(Z_new, bool):
            # Z_new = torch.unsqueeze(torch.arange(-10,10,0.02),1)
            # generate the Z_new from a fiexed normal distribution
            mean = torch.zeros(Z_dim)
            covariance_matrix = torch.eye(Z_dim)
            multivariate_normal = MultivariateNormal(mean, covariance_matrix)
            Z_new = multivariate_normal.sample((10000,)) 
            pass
        Z_new = Z_new.cpu().detach()

        
        T_est = torch.sort(self.T[self.D==1])
        base_line = torch.exp(-torch.cumsum(self.hratio[T_est.indices], dim=0)).cpu().detach()
        T_est = T_est.values.cpu().detach()
        
        
        # plot the result
        plt.figure(figsize=(9,3))
        plt.subplot(1,2,1)

        line1, = plt.plot(T_est,base_line,color='red')
        if Baseline:
            line2, = plt.plot(T_est, Baseline(T_est),color='black')
            plt.legend(handles=[line1,line2],labels=['Estimated baseline','True baseline'],loc='upper right',fontsize=8)
        
        plt.xlabel('T')
        plt.ylabel('S(t)=Pr(T>t)')
        plt.title('Baseline function plot')
        
        
        plt.subplot(1,2,2)
        theta_true = theta(Z_new).detach()-torch.mean(theta(Z_new).detach()) 
        plt.scatter(theta_true,
                    theta_hat(Z_new.to(device)).detach().cpu(),color='blue',s=0.5)
        a = torch.tensor([torch.min(theta_true), torch.max(theta_true)]).float()
        plt.plot(a,a,color='red')
         
        plt.xlabel('theta(Z)')
        plt.ylabel('theta_hat(Z)')
        plt.title('Partial linear estimation plot')
        if save_fig:
            plt.savefig(save_fig)

        plt.subplots_adjust(wspace=0.5)
        plt.show()
        
        pass
    pass


    def plot_DeepSurv(self, Baseline = False, theta = False, Z_new = False,save_fig=False):
        device=self.device
        theta_hat = self.theta
        if isinstance(Z_new, bool):
            Z_new = torch.unsqueeze(torch.arange(torch.quantile(self.Z,q=0.025),torch.quantile(self.Z,q=0.975),0.01),1)
            pass
        Z_new = Z_new.cpu().detach()

        
        T_est = torch.sort(self.T[self.D==1])
        base_line = torch.exp(-torch.cumsum(self.hratio[T_est.indices], dim=0)).cpu().detach()
        T_est = T_est.values.cpu().detach()
        
        
        # start plot
        plt.figure(figsize=(9,3))
        
        plt.subplot(1,2,1)
        line1, = plt.plot(T_est,base_line,color='red')
        if Baseline:
            line2, = plt.plot(T_est, Baseline(T_est),color='black')
            plt.legend(handles=[line1,line2],labels=['Estimated baseline','True baseline'],loc='upper right',fontsize=8)
        
        plt.xlabel('T')
        plt.ylabel('S(t)=Pr(T>t)')
        plt.title('Baseline function plot')
        
        
        plt.subplot(1,2,2)
        line1, = plt.plot(Z_new.detach(),theta_hat(Z_new.to(device)).detach().cpu(),color='red')
        if theta:
            line2, = plt.plot(Z_new.detach(), 
                              theta(Z_new).detach()-torch.mean(theta(Z_new).detach()), 
                              color='black')
            plt.legend(handles=[line1,line2],labels=['Estimated theta(Z)','True theta(Z)'],loc='upper right',fontsize=8)
        
        plt.xlabel('Z')
        plt.ylabel('theta(Z)')
        plt.title('Partial linear part plot')
        if save_fig:
            plt.savefig(save_fig)

        plt.subplots_adjust(wspace=0.5)
        plt.show()
        
        pass
    pass




# Define the class to record the information in E[X|Z,T>t]
class Model_X_ZT:
    def __init__(self,T_interval,model,device):
        self.T_interval = T_interval.to(device)
        self.model=model
        self.device=device
        pass

    def plot2(self,sim_data,E_XZ,Z_new = False,save_fig=False):
        Z=sim_data.Z
        Z_dim=Z.shape[1]
        
        if isinstance(Z_new, bool):
            # Z_new = torch.unsqueeze(torch.arange(-10,10,0.02),1)
            mean = torch.zeros(Z_dim)
            covariance_matrix = torch.eye(Z_dim)
            multivariate_normal = MultivariateNormal(mean, covariance_matrix)
            Z_new = multivariate_normal.sample((10000,)) 
            pass
        Z_new = Z_new.cpu().detach()
        T_interval = self.T_interval.cpu().detach()
        model = self.model[0].cpu()
        
        T = sim_data.T.cpu().detach()
        X = sim_data.X.cpu().detach()
        Z = sim_data.Z.cpu().detach()
        
        plt.figure(figsize=(min(3*X.shape[1],16),3))
        exz =  E_XZ(Z_new, X_dim=X.shape[1]).clone().detach()
        for i in range(X.shape[1]):
            plt.subplot(1,X.shape[1],i+1)
            plt.scatter(exz[:,i], model(Z_new)[:,i].cpu().detach(), s=0.5,color='blue')
            a = torch.tensor([torch.min(exz[:,i]), torch.max(exz[:,i])]).float()
            plt.plot(a,a,color='red')
            plt.xlabel('E[X(%d)|Z]'%(i+1))
            plt.ylabel('E_hat[X(%d)|Z]'%(i+1))
            pass
        
        plt.subplots_adjust(wspace=0.5)
        if save_fig:
            plt.savefig(save_fig)
        plt.show()
        
    
    def plot(self,sim_data,Z_new = False,save_fig=False):
        Z=sim_data.Z
        if isinstance(Z_new, bool):
            Z_new = torch.unsqueeze(torch.arange(torch.quantile(Z,q=0.025),torch.quantile(Z,q=0.975),0.01),1)
            pass
        Z_new = Z_new.cpu().detach()
        T_interval = self.T_interval.cpu().detach()
        model = self.model
        
        T = sim_data.T.cpu().detach()
        X = sim_data.X.cpu().detach()
        Z = sim_data.Z.cpu().detach()
        
        
        K = len(T_interval)
        plt.figure(figsize=(min(2*K,16),2*X.shape[1]))
        for k in range(K):
            for i in range(X.shape[1]):
                plt.subplot(X.shape[1], K, k+K*i+1)
                select = torch.squeeze(T)>= T_interval[k]
                plt.scatter(Z[select],X[select,i],s=0.3)
                mod = model[k].cpu()
                plt.plot(Z_new, mod(Z_new)[:,i].detach(), color='red')
                plt.title('E[X(%d)|Z,T>%.2f]'%(i+1, T_interval[k].cpu()))
                pass
            pass

        plt.subplots_adjust(hspace=0.5)
        if save_fig:
            plt.savefig(save_fig)
        plt.show()
        
        pass
    
    def predict(self,T_new,Z_new, plot_pic = False):
        device =self.device
        T_new = T_new.to(device)
        Z_new = Z_new.to(device)
        T_interval = self.T_interval.to(device)
        model = self.model
        
        n=len(T_new)
        # 每行求一个最小值
        RowMin = torch.min(torch.abs(T_new-T_interval),1)
        index = RowMin.indices
        temp = RowMin.values

        Weight = torch.unsqueeze(torch.eq(torch.tile(torch.unsqueeze(temp,1),[1,len(T_interval)]), 
                                          torch.abs(T_new-T_interval)).to(torch.float32),2)
        for i in range(len(T_interval)):
            mod = model[i].to(device)
            XZ_values = mod(Z_new)
            if i==0:
                fit_XZ_values = torch.zeros([n,XZ_values.shape[1],len(T_interval)]).to(device)
                pass
            fit_XZ_values[:,:,i]=XZ_values
            pass

        fit_XZ_values = torch.squeeze(torch.bmm(fit_XZ_values,Weight),2)
        if plot_pic:
            plt.figure(figsize=(12,2))
            for i in range(len(T_interval)):
                select= (index==i).cpu()
                plt.subplot(1,len(T_interval),i+1)
                plt.scatter(Z_new[select,0].cpu().detach(),fit_XZ_values[select,0].cpu().detach(),s=0.3)
            plt.show()
            pass
        
        return fit_XZ_values
    
    def predict_cross(self,T_new,Z_new,plot_pic=False):
        
        device = self.device
        T_new=T_new.to(device)
        Z_new=Z_new.to(device)

        # T_temp = kron(T_new,torch.ones([len(Z_new),1]).to(device))
        # Z_temp = kron(torch.ones([len(T_new),1]).to(device),Z_new)
        # print(T_temp.shape)
        T_temp = kron(torch.ones([len(Z_new),1]).to(device),T_new)
        Z_temp = kron(Z_new,torch.ones([len(T_new),1]).to(device))
        res = self.predict(T_temp,Z_temp).reshape(len(Z_new),len(T_new),-1)
        
        
        # res=torch.zeros([len(Z_new),len(T_new),2]).to(device)
        # for i in range(len(Z_new)):
        #     Z_temp = Z_new[i] * torch.ones((len(T_new),1)).to(device)
        #     res[i,:,:] =  self.predict(T_new,Z_temp)
        #     pass

        if plot_pic:
            fig = plt.figure(figsize=(15,5))

            for i in range(res.shape[2]):
            
                ax = fig.add_subplot(1,res.shape[2],i+1, projection='3d')
            
                surf = ax.plot_surface(Z_new.cpu().detach().numpy(), 
                                   T_new.cpu().detach().numpy(), 
                                   res[:,:,i].cpu().detach().numpy(),
                                   cmap='viridis')

                # surf = ax.plot_surface(T_new.cpu().detach().numpy(), 
                #    Z_new.cpu().detach().numpy(), 
                #    torch.t(res[:,:,i]).cpu().detach().numpy(),
                #    cmap='viridis')
                ax.set_xlabel('Z')
                ax.set_ylabel('t')
                ax.set_zlabel('E[X|Z%d,T>t]'%i)
            plt.show()
            
        
        return res    
    
    pass

# Defining the neural network for computing the variance of nuisance estimator 
class CustomNet(nn.Module):
    def __init__(self, h_dims, g_dims, weight_decay=0,random_start=True):
        super(CustomNet, self).__init__()
        # construct the neural network according to the layer_dims h_dims and g_dims
        self.h = self._create_fc_net(h_dims,random_start=random_start)
        self.g = self._create_fc_net(g_dims,random_start=random_start)
        self.weight_decay = weight_decay

    def _create_fc_net(self, layer_dims,random_start=True):
        layers = []
        for i in range(len(layer_dims) - 2):
            net_append= nn.Linear(layer_dims[i], layer_dims[i+1])
            if not random_start:
                net_append.weight.data.fill_(0)
                net_append.bias.data.fill_(0)
            layers.append(net_append)
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        net = nn.Sequential(*layers)   
        return net

    def forward(self, T, Z):
        h_out = self.h(T)
        g_out = self.g(Z)
        return h_out, g_out

    def regularization_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.norm(param)**2
        return self.weight_decay * reg_loss




# kronecker product
def kron(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

# Partial likelihoood loss function
def loss_Cox(time, status, eta):
    T_mat = (torch.unsqueeze(time,1)<=time).to(torch.float32)
    loss = -torch.mean(status * (eta - torch.log(T_mat @ torch.exp(eta))))
    return loss
    pass

# compute the loss function
def custom_loss(D, X, h_out, g_out, reg_loss=0):
    return torch.mean(torch.sum(D*(X - h_out - g_out)**2,dim=1)) + reg_loss



class JointDataset(Dataset):
    def __init__(self, T, D, X, Z):
        self.T = T
        self.D = D
        self.X = X
        self.Z = Z

    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        return self.T[idx,:], self.D[idx,:],self.X[idx,:],self.Z[idx,:]
    pass



'''
N: sampling number,
beta: linear parameter of interst in hazard function,
theta: nuisance parameter of the hazard function,
device: the device to run the simulation,
E_XZ: the function to generate the E[X|Z],
dist_Z: the function to generate the distribution of Z,
censor_ave: the average censoring rate,
shape: shape parameter of the Weibull distribution,
scale: scale parameter of the Weibull distribution,
cute: the upper bound of the Weibull distribution,
X_type: the type of X, 'binomial' or 'normal',
'''
def sim_survival(N,beta,theta,device,
                 E_XZ=False, dist_Z=False, censor_ave=1,
                 shape=2,scale=1,cut=1e8,X_type='binomial'):
    
    if not dist_Z:
        def dist_Z(N,Z_dim,rho_Z=0):
            mean = torch.zeros(Z_dim)
            covariance_matrix = torch.zeros((Z_dim, Z_dim))
            
            for i in range(Z_dim):
                for j in range(Z_dim):
                    covariance_matrix[i, j] = rho_Z**np.abs(i-j)
            multivariate_normal = MultivariateNormal(mean, covariance_matrix)
            Z = multivariate_normal.sample((N,))
            
            normal_dist = Normal(mean, 1)
            Z = normal_dist.cdf(Z)
            return Z
    
    if not E_XZ:
        def E_XZ(Z,X_dim):
            np.random.seed(204)
            A = torch.tensor(np.random.normal(0,1,size=( 3*Z.shape[1], X_dim))).float()
            eta = torch.cat((torch.sin(Z), Z,torch.sin(Z*3)),1) @ A
            return torch.sigmoid(feature_normalize(eta))
            pass

    
    Z=dist_Z(N)
    if X_type=='binomial':
        binomial_dist = torch.distributions.Binomial(total_count=1, probs=E_XZ(Z,X_dim=len(beta)))
        X = binomial_dist.sample()
    elif X_type=='gaussian':
        X = E_XZ(Z,X_dim=len(beta)) + torch.normal(0,0.5,size=(N,len(beta)))
    else:
        return 0


    # X = torch.cat((-torch.exp(torch.abs(Z)), Z,0.5*torch.sin(Z)),1) @ A 
    lin_pred = X @ torch.unsqueeze(beta,1) + theta(Z) - torch.mean(theta(Z))
    
    # generate Weilull distribution
    U = torch.unsqueeze(torch.rand(N),1)
    # change scale properly
    T = scale * (-torch.log(U) / torch.exp(lin_pred))**(1 / shape)
    censoring_time = torch.minimum(torch.unsqueeze(-torch.log(torch.rand(N))*censor_ave,1),torch.tensor(cut))
    # observed times are min of true times and censoring times
    time = np.minimum(T, censoring_time)
    # status indicator: 1 if event occurred, 0 if censored
    status = (T <= censoring_time).to(torch.float32)  
    
    return DeepSurvData(time,status,X,Z,device)



# fit Deep survival Model
# Data_loader should contain (T,D,X,Z) from the partial linear model，
# Output: beta_hat and theta_hat
def fit_CoxPH(sim_data, device, model=False,Epoch=2000,lr=1e-3, Early_stop=True,
              print_state=True,Test_set=False):
    sim_data = sim_data.to(device)
    T = sim_data.T
    D = sim_data.D
    X = sim_data.X
    Z = sim_data.Z
    
    data_loader = DataLoader(JointDataset(T,D,X,Z), batch_size=6000, shuffle=True, drop_last=False)
    # beta_hat =  torch.zeros(4,1,requires_grad=True).to(device)
    if not model:        
        theta = nn.Sequential(
            nn.Linear(Z.shape[1],30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.Sigmoid(),
            nn.Linear(30,1),
        ).to(device)
    else:
        theta = copy.deepcopy(model).to(device)
    
    beta_hat = nn.Linear(X.shape[1],1,bias=False).to(device)    
    optimizer_theta = torch.optim.Adam(theta.parameters(),lr=lr)
    optimizer_beta = torch.optim.Adam(beta_hat.parameters(),lr=lr)
    loss_test_rec=100000
    
    for epoch in range(Epoch):
        
        for index, item in enumerate(data_loader):
            # print('batch:',index,'\nT:{},\nD:{},\nX:{}'.format(item[0],item[1],item[2]))
            T_ = item[0]
            D_ = item[1]
            X_ = item[2]
            Z_ = item[3]

            loss = loss_Cox(torch.squeeze(T_,1),D_,theta(Z_)+beta_hat(X_))
            loss.backward()
            optimizer_theta.step()
            optimizer_theta.zero_grad()
            
            optimizer_beta.step()
            optimizer_beta.zero_grad()

            pass
        
        if epoch%10 ==0:
            loss_test=0
            if Test_set:
                Test_set=Test_set.to(device)
                T_test=Test_set.T
                D_test=Test_set.D
                X_test=Test_set.X
                Z_test=Test_set.Z
                loss_test = loss_Cox(torch.squeeze(T_test,1),D_test,theta(Z_test)+beta_hat(X_test))
            
            if print_state:
                if Test_set:
                    print('epoch:',epoch,'Train_loss:%.5f'%loss.cpu().detach().numpy(),end=', ')
                    print('epoch:',epoch,'Test_loss:%.5f'%loss_test.cpu().detach().numpy())
                else:
                    print('epoch:',epoch,'Train_loss:%.5f'%loss.cpu().detach().numpy())
                    pass
                    
            if Test_set and Early_stop and epoch>40 and loss_test>loss_test_rec:
                break
            loss_test_rec=loss_test 
                    
        pass


    optimizer_beta.param_groups[0]['lr'] = 0.02
    for epoch in range(20):
        for index, item in enumerate(data_loader):
            # print('batch:',index,'\nT:{},\nD:{},\nX:{}'.format(item[0],item[1],item[2]))
            T_ = item[0]
            D_ = item[1]
            X_ = item[2]
            Z_ = item[3]

            loss = loss_Cox(torch.squeeze(T_,1),D_,theta(Z_)+beta_hat(X_))
            loss.backward()
            
            optimizer_beta.step()
            optimizer_beta.zero_grad()

            pass
 
            
            pass
        pass
 
    return torch.squeeze(beta_hat.weight.data,0), (lambda x: theta(x)-torch.mean(theta(Z))) #,torch.inverse(Beta_hessian)/len(X)
    pass


# Data_loader should contain (T,D,X,Z) from the partial linear model 
# fit E[X|Z,T>=MinT]
def fit_XZ(sim_data,device, T_interval, model_XZ= False, Epoch = 2000,lr=1e-3, Early_stop=True,
           loss_fn = nn.MSELoss(),print_state = True,Test_set=False):
    sim_data=sim_data.to(device)
    T=sim_data.T
    D=sim_data.D
    X=sim_data.X
    Z=sim_data.Z
    
    data_loader = DataLoader(JointDataset(T,D,X,Z),
                     batch_size=5000, shuffle=True, drop_last=False)
    if not model_XZ:
        model_XZ_single = nn.Sequential(
            nn.Linear(Z.shape[1],60),
            nn.ReLU(),
            nn.Linear(60,60),
            nn.Sigmoid(),
            nn.Linear(60,60),
            nn.ReLU(),
            nn.Linear(60,X.shape[1]),
        ).to(device)
    else:
        # model_XZ = [copy.deepcopy(model_XZ).to(device),]*len(T_interval) 
        model_XZ_single = copy.deepcopy(model_XZ).to(device)
        pass

    
    model_XZ = list()
    optimizer_model_XZ=list()
    for i in range(len(T_interval)):
        model_XZ.append(copy.deepcopy(model_XZ_single).to(device))
        optimizer_model_XZ.append(torch.optim.Adam(model_XZ[i].parameters(),lr=lr))
        pass
    loss_rec=100000

    for epoch in range(Epoch):
        for index, item in enumerate(data_loader):
            # print('batch:',index,'\nT:{},\nX:{},\nY:{}'.format(item[0],item[1],item[2]))
            T = torch.squeeze(item[0],1)
            X = item[2]
            Z = item[3]
            # print(Z.shape)

            loss=0
            for i in range(len(T_interval)):
                subset = T>= T_interval[i]
                X_select = X[subset,:]
                Z_select = Z[subset,:]
                loss = loss + loss_fn(model_XZ[i](Z_select),X_select)
            
            loss.backward()
            
            for i in range(len(T_interval)):
                optimizer_model_XZ[i].step()
                optimizer_model_XZ[i].zero_grad()
            
            pass
        
        # jump out the loop if the loss is increasing
        if epoch%10 ==0:
            # compute the loss on the test set
            loss_test=0
            if Test_set:
                Test_set=Test_set.to(device)
                T_test=torch.squeeze(Test_set.T,1)
                D_test=Test_set.D
                X_test=Test_set.X
                Z_test=Test_set.Z
                for i in range(len(T_interval)):
                    subset = T_test>= T_interval[i]
                    X_test_select = X_test[subset,:]
                    Z_test_select = Z_test[subset,:]
                    loss_test = loss_test + loss_fn(model_XZ[i](Z_test_select),X_test_select)
                    pass
                    
            # if print the state if print_state is True
            if print_state:
                if Test_set:
                    print('epoch:',epoch,'Train_loss:%.5f'%loss.cpu().detach().numpy(),end=', ')
                    print('epoch:',epoch,'Test_loss:%.5f'%loss_test.cpu().detach().numpy())
                else:
                    print('epoch:',epoch,'Train_loss:%.5f'%loss.cpu().detach().numpy())
                    pass

            # if the loss is increasing, then stop the training        
            if Test_set and Early_stop and epoch>40 and loss_test>loss_test_rec:
                break
            loss_test_rec=loss_test 
            
            pass
        pass
    
    return model_XZ
    pass


def fit_XZ_complete(sim_data, T_interval, device, model=False, Epoch=2000,lr=1e-3, Early_stop=True,
                    loss_fn = nn.MSELoss(),print_state=True,Test_set=False):
    sim_data =sim_data.to(device)
    T_interval = T_interval.to(device)
    model_XZ = fit_XZ(sim_data,device=device,T_interval = T_interval,model_XZ=model,
                      Epoch=Epoch, lr=lr,Early_stop=Early_stop,loss_fn=loss_fn,
                      print_state=print_state,Test_set=Test_set)
    
    return Model_X_ZT(T_interval,model_XZ,device)



def fit_Nuisance(sim_data,subset,device,accuracy, Early_stop=True,
                 loss_XZ=nn.MSELoss(),Test_set=False,
                 model_XZ=False,model_theta=False, 
                 Epoch_XZ=2000, Epoch_theta=2000,
                 lr_XZ=1e-3, lr_theta=1e-3, print_state=True):
    sim_data = sim_data.to(device)
    T = sim_data.T[subset,:]
    D = sim_data.D[subset,:]
    X = sim_data.X[subset,:]
    Z = sim_data.Z[subset,:]
    # data_loader = DataLoader(JointDataset(T,D,X,Z),shuffle=True, batch_size=2000,drop_last=False)
    
    if print_state:
        print('train theta(Z)')
        
    # beta_est,theta_est, beta_var = fit_CoxPH(sim_data,model=model_theta,device=device)
    beta_est,theta_est = fit_CoxPH(sim_data,model=model_theta, Early_stop=Early_stop,
                                   Epoch = Epoch_theta ,lr=lr_theta,
                                   device=device,print_state=print_state,Test_set=Test_set)
    # print(torch.sqrt(torch.diag(beta_var)))
    deepSurv = DeepSurv(sim_data,beta_est,theta_est,device=device)
    
    if print_state:
        print('train E[X|Z,T>t]')
        
    T_interval = torch.quantile(T,torch.arange(0,0.81,accuracy).to(device))
    model_XZ = fit_XZ_complete(sim_data ,T_interval,device=device,model=model_XZ, Early_stop=Early_stop,
                               Epoch=Epoch_XZ,lr=lr_XZ, print_state=print_state,loss_fn=loss_XZ,Test_set=Test_set)
    
    return beta_est, deepSurv, model_XZ

def psi1(beta, sim_data, subset, deepSurv, model_XZ,need_details = False):
    beta=torch.unsqueeze(beta,1)
    
    T = sim_data.T[subset,:]
    D = sim_data.D[subset,:]
    X = sim_data.X[subset,:]
    Z = sim_data.Z[subset,:]
    
    X_hat_self = model_XZ.predict(T,Z)
    
    if need_details:
        return torch.multiply(X -  X_hat_self, D * torch.exp(-X@beta)).detach()
    else:
        res = torch.t(X -  X_hat_self)@ (D * torch.exp(-X@beta))
        return res.detach()
        pass
    pass


def psi2(sim_data, subset, deepSurv, model_XZ,device):
    T = sim_data.T[subset,:]
    D = sim_data.D[subset,:]
    X = sim_data.X[subset,:]
    Z = sim_data.Z[subset,:]

    
    X_hat_cross = model_XZ.predict_cross(torch.unsqueeze(deepSurv.T_D1,1),Z)
    # print(X_hat_cross.shape)
    res =torch.zeros([X.shape[1],1]).to(device)
    mat_detail = torch.zeros([len(T),X.shape[1]]).to(device)
    for i in range(X.shape[1]):
        
        Weight = (torch.tile(torch.unsqueeze(X[:,i],1),[1,len(deepSurv.T_D1)]) - X_hat_cross[:,:,i]) *(T>=deepSurv.T_D1).to(torch.float32)  
        res[i] = torch.t(torch.exp(deepSurv.theta(Z)))@ Weight @ torch.unsqueeze(deepSurv.hratio,1)
        mat_detail[:,i] = torch.squeeze(torch.exp(deepSurv.theta(Z)) * (Weight@torch.unsqueeze(deepSurv.hratio,1)),1).detach()
        pass
    
    return res.detach(), mat_detail




def psi_grad(beta, sim_data, subset, deepSurv, model_XZ, need_details = False):
    beta=torch.unsqueeze(beta,1)
    
    T = sim_data.T[subset,:]
    D = sim_data.D[subset,:]
    X = sim_data.X[subset,:]
    Z = sim_data.Z[subset,:]
    
    X_hat_self = model_XZ.predict(T,Z)
    
    res = -torch.t(X -  X_hat_self)@ torch.diag(torch.squeeze(D * torch.exp(-X@beta),1)) @ X
    
    return res.detach()

'''
input:
    sim_data: survival data in partial linear model,
    device: cuda to use,
    accuracy: number of pieces of T_interval,
    loss_XZ: the loss function to train E[X|Z,T>t],
    Test_set: it is used to compute the loss on testing set,
    
    model_XZ: the neural network to fit E[X|Z,T>t],
    Epoch_XZ: the epoch needed to fit E[X|Z,T>t],
    lr_XZ: learning rate to train E[X|Z,T>t],
    model_theta: the neural network to fit theta(Z),
    
    Epoch_theta: the epoch needed to fit theta(Z),
    lr_theta: learning rate to train theta(Z),
    print_state: print the training process or not.
    
output:
    beta_DB: Double robust estimator,
    var_est: variance estimator for beta_DB,    
'''
def DML_CoxPH(sim_data, device, accuracy=0.2, Early_stop=True,
              loss_XZ=nn.MSELoss(), Test_set=False,
              model_XZ=False,model_theta=False,
              Epoch_XZ=2000,Epoch_theta=2000,
              lr_XZ=1e-3, lr_theta=1e-3,
              print_state = True):
    sim_data=sim_data.to(device)

    # sample=np.random.permutation(np.arange(len(sim_data.T))).reshape(2,int(len(sim_data.T)/2))
    sample = np.random.permutation(np.arange(len(sim_data.T)))
    sample = np.array_split(sample,2)
    
    # sample1 = sample[0,:]
    # sample2 = sample[1,:]
    sample1 = sample[0]
    sample2 = sample[1]
    
    beta_est1, deepSurv1, model_XZ1 = fit_Nuisance(sim_data,subset = sample1,device=device, accuracy=accuracy, 
                                                   loss_XZ=loss_XZ, Test_set=Test_set, Early_stop=Early_stop,
                                                   model_XZ=model_XZ, model_theta=model_theta,
                                                   Epoch_XZ=Epoch_XZ,Epoch_theta=Epoch_theta,
                                                   lr_XZ=lr_XZ, lr_theta=lr_theta,
                                                   print_state=print_state)
    beta_est2, deepSurv2, model_XZ2 = fit_Nuisance(sim_data,subset = sample2,device=device, accuracy=accuracy, 
                                                   loss_XZ=loss_XZ, Test_set=Test_set, Early_stop=Early_stop,
                                                   model_XZ=model_XZ, model_theta=model_theta, 
                                                   Epoch_XZ=Epoch_XZ,Epoch_theta=Epoch_theta,
                                                   lr_XZ=lr_XZ, lr_theta=lr_theta,
                                                   print_state=print_state)
    
    psi_constant1, psi_mat1_constant = psi2(sim_data, sample1,deepSurv2,model_XZ2,device) 
    psi_constant2, psi_mat2_constant = psi2(sim_data, sample2,deepSurv1,model_XZ1,device)
    psi_constant = (psi_constant1 + psi_constant2)
    
#     beta_naive=(beta_est1+beta_est2)/2
    beta_DB = torch.zeros([sim_data.X.shape[1]]).to(device)
    for i in range(100):
        psi_para = (psi1(beta_DB,sim_data, sample1,deepSurv2,model_XZ2) + psi1(beta_DB,sim_data, sample2,deepSurv1,model_XZ1))
        psi_total = psi_para - psi_constant
        psi_grad_total = psi_grad(beta_DB, sim_data, sample1,deepSurv2,model_XZ2) + psi_grad(beta_DB, sim_data, sample2,deepSurv1,model_XZ1)
        # print(i)
        # print(psi_grad_total)
        # print(torch.inverse(psi_grad_total))
        # print(psi_total)
        beta_DB = beta_DB - torch.squeeze(torch.inverse(psi_grad_total) @ psi_total, 1)
        if torch.norm(psi_total)<1e-5:
            break
            pass
        pass
    # print('over')
    psi_mat1_par = psi1(beta_DB,sim_data, sample1,deepSurv2,model_XZ2,need_details=True)
    psi_mat2_par = psi1(beta_DB,sim_data, sample2,deepSurv1,model_XZ1,need_details=True)
    psi_mat1 = psi_mat1_par - psi_mat1_constant
    psi_mat2 = psi_mat2_par - psi_mat2_constant
    psi_square = torch.t(psi_mat1) @ psi_mat1 + torch.t(psi_mat2) @ psi_mat2
    var_est = torch.inverse(psi_grad_total) @ psi_square @ torch.inverse(psi_grad_total)


    return beta_DB, var_est, beta_est1, deepSurv1, model_XZ1, beta_est2, deepSurv2, model_XZ2



'''
sim_data: survival data in partial linear model,
net: the neural network to fit the variance estimator,
device: cuda to use,
lr: learning rate to train the variance estimator,
Epoch: the epoch needed to train the variance estimator,
print_state: print the training process or not.
Epoch: the epoch needed to train the variance estimator,
Early_stop: early stop or not.
print_state: print the training process or not.
'''
def fit_variance(sim_data, net, device, lr=1e-2,Epoch=20000,Early_stop = True, print_state=True):
    sim_data=sim_data.to(device)
    net=net.to(device)

    
    T = feature_normalize(sim_data.T)
    Z = feature_normalize(sim_data.Z)
    D = sim_data.D
    X = sim_data.X
    
    
    dataset = JointDataset(T,D,X,Z)
    # divede the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=3200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3200, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # start training
    val_loss_rec=10000
    for epoch in range(Epoch):
        # training parameters
        net.train()
        for T, D, X, Z in train_loader:
            # compute the loss and gradients
            h_out, g_out = net(T, Z)
            reg_loss = net.regularization_loss()
            loss = custom_loss(D, X, h_out, g_out, reg_loss)
    
            # backward
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
    
        net.eval()
    
        # print the training process if needed
        if (epoch + 1) % 10 == 0:
            # validation
            val_loss = 0
            with torch.no_grad():
                for T, D, X, Z in val_loader:
                    h_out, g_out = net(T, Z)
                    reg_loss = net.regularization_loss()
                    val_loss += custom_loss(D, X, h_out, g_out, reg_loss).item()
                    pass
            if print_state:
                print('Epoch %d, Train Loss: %.6f, Val Loss: %.6f'%(epoch+1,loss.item(), val_loss / len(val_loader)))
            
            if val_loss>val_loss_rec and epoch>30 and Early_stop:
                break
        
            val_loss_rec=val_loss
            pass

    h_hat, g_hat = net(feature_normalize(sim_data.T), feature_normalize(sim_data.Z))
    fit_sample = sim_data.D*(sim_data.X - h_hat - g_hat)
    
    return torch.inverse(fit_sample.t() @ fit_sample).detach()




