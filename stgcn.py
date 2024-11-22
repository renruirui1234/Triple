import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, kernel_size),padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, kernel_size),padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3, kernel_size),padding=(1,1))
        self.pool1=nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2),padding=(1,1))
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)#kernel为3,输入为2，输出为64，所以我们修改的时候先按照输入channel为1吧，因为只有包长度一个，或者可以加上上下行，但是感觉上下行得多流合并
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))#这部分应该就是GLU
        out = F.relu(temp + self.conv3(X))
        out=self.pool1(out)
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out



class BatchNormBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, num_features):
        super(BatchNormBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)

    def forward(self, X):
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 2, 1, 3)#kernel为3,输入为2，输出为64，所以我们修改的时候先按照输入channel为1吧，因为只有包长度一个，或者可以加上上下行，但是感觉上下行得多流合并
        out=self.batch_norm(X)
        out = out.permute(0, 2, 1, 3)
        return out




class SpatialBlock(nn.Module):
    """
        Neural network block that applies a temporal convolution on each node in
        isolation, followed by a graph convolution, followed by another temporal
        convolution on each node.
        """

    def __init__(self,spatial_channels, out_channels,
                 num_nodes):

        super(SpatialBlock, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, X.permute(1, 0, 2, 3)])
        lfs_cpu = np.sum(lfs.detach().cpu().numpy(),axis=(0,2,3))

        t2 = F.leaky_relu(torch.matmul(lfs, self.Theta1))
        Theta1_cpu=self.Theta1.detach().cpu().numpy()
        t2_cpu = np.sum(t2.detach().cpu().numpy(), axis=(0, 2, 3))

        return self.batch_norm(t2)
        # return t3





class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 embedding_num):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.T_block1 = TimeBlock(in_channels=num_features, out_channels=num_features*2)
        self.batch_norm1 = BatchNormBlock(num_features=int(2))

        self.T_block2 = TimeBlock(in_channels=num_features*2, out_channels=num_features*4)
        self.batch_norm2 = BatchNormBlock(num_features=int(2))

        self.T_block3 = TimeBlock(in_channels=num_features*4, out_channels=num_features*4)
        self.batch_norm3 = BatchNormBlock(num_features=int(2))

        self.T_block4 = TimeBlock(in_channels=num_features * 4, out_channels=num_features * 2)
        self.batch_norm4 = BatchNormBlock(num_features=int(2))

        self.embedding_num=embedding_num
        self.num_features=num_features
        self.S_block1=SpatialBlock(spatial_channels=num_features, out_channels=num_features*4,
                 num_nodes=num_nodes)


        self.S_block2=SpatialBlock(spatial_channels=num_features, out_channels=num_features,
                 num_nodes=num_nodes)

        self.fully = nn.Linear(2*num_features * num_nodes,
                               num_features)
        # self.fully = nn.Linear(2*num_features*num_nodes,
        #                        num_features)


    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.T_block1(X)
        out1=self.batch_norm1(out1)
        out1_cpu = out1.detach().cpu().numpy()
        out2 = self.T_block2(out1)
        out3 = self.batch_norm2(out2)
        out2_cpu = out2.detach().cpu().numpy()
        # out4 = self.T_block3(out3)
        # out4 = self.batch_norm3(out4)
        # out3_cpu = out3.detach().cpu().numpy()
        # out4=self.last_temporal(out3)
        # out4=out3.reshape((out3.shape[0], -1))
        # out4 = self.fully(out4)
        # out4_cpu=out4.detach().cpu().numpy()

        out4=self.S_block1(out3,A_hat)
        out4_cpu = np.sum(out4.detach().cpu().numpy(),axis=(0,2,3))
        # print(out4_cpu)

        # out5 = out4.reshape((out4.shape[0], -1))


        out5 = self.S_block2(out4, A_hat)#由于值都是负数，感觉是因为batch_norm的原因
        # out5_cpu = np.sum(out5.detach().cpu().numpy(), axis=(0, 2, 3))
        out5 = out5.reshape((out5.shape[0], -1))
        out6 = self.fully(out5)
        out6_cpu = out6.detach().cpu().numpy()


        # result=torch.softmax(out6,dim=1)
        # result_cpu=result.detach().cpu().numpy()
        # result_cpu_1=np.sum(result_cpu_1,axis=1)
        return out6


