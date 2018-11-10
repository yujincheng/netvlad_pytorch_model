
import argparse
import torch
import collections
import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch.nn.functional as F
import glob
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as scid
import scipy.signal as scisig
import time
import unittest
from sklearn import preprocessing
from scipy.spatial.distance import pdist



class Vd16_pitts30k_conv5_3(nn.Module):
    class NetVLAD(nn.Module):
        """NetVLAD layer implementation"""
        def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                    normalize_input=True):
            """
            Args:
                num_clusters : int
                    The number of clusters
                dim : int
                    Dimension of descriptors
                alpha : float
                    Parameter of initialization. Larger value is harder assignment.
                normalize_input : bool
                    If true, descriptor-wise L2 normalization is applied to input.
            """
            super().__init__()
            self.num_clusters = num_clusters
            self.dim = dim
            self.alpha = alpha
            self.normalize_input = normalize_input
            self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), padding=(0, 0), bias=False)
            self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        #     self._init_params()

        # def _init_params(self):
        #     self.conv.weight = nn.Parameter(
        #         (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        #     )
        #     self.conv.bias = nn.Parameter(
        #         - self.alpha * self.centroids.norm(dim=1)
        #     )

        def forward(self, x):
            N, C = x.shape[:2]

            if self.normalize_input:
                x = F.normalize(x, p=2, dim=1)  # across descriptor dim

            # soft-assignment
            soft_assign = self.conv(x).view(N, self.num_clusters, -1)
            soft_assign = F.softmax(soft_assign, dim=1)

            x_flatten = x.view(N, C, -1)

            # calculate residuals to each clusters
            residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) + \
                self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign.unsqueeze(2)
            vlad = residual.sum(dim=-1)

            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
            vlad = vlad.permute(0,2,1).contiguous()
            vlad = vlad.view(x.size(0), -1)  # flatten
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize


            return vlad


    def __init__(self):
        super().__init__()

        self.meta = {'mean': [123.68000000000028, 116.77899999999951, 103.93899999999951],
                     'std': [1, 1, 1]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.relu1_2 = nn.ReLU()
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.relu2_2 = nn.ReLU()
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.relu3_3 = nn.ReLU()
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.relu4_3 = nn.ReLU()
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.net_vlad = self.NetVLAD(num_clusters=64, dim=512, alpha=1.0)
        self.WPCA= nn.Linear(64*512,4096)

        # self.preL2 = nn.LocalResponseNorm(1024,alpha = 1024*1, beta = 0.5, k = 1e-12)

        # self.netvlad_layer = nn.Sequential(collections.OrderedDict([
        #     ('convsoft',nn.Conv2d(512,64,kernel_size=[1, 1], stride=(1, 1), padding=(0, 0))),
        #     ('softmax1',nn.Softmax2d()),
        #     ('onlybias',self.OnlyBias(512,64))
        # ]))

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.pool1(x3)
        x5 = self.relu1_2(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.pool2(x8)
        x10 = self.relu2_2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.pool3(x15)
        x17 = self.relu3_3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.pool4(x22)
        x24 = self.relu4_3(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.net_vlad(x29)
        x31 = self.WPCA(x30)
        x32 = F.normalize(x31,p=2,dim=1)
        # x_preL2 = self.preL2(x29)
        # x_assgn =  self.netvlad_layer(x_preL2)
        return x32

def trans_main():
    use_cuda = True
    pt_net = Vd16_pitts30k_conv5_3()
    # matfolder = '../matlab_trained/matlab_param/'
    model = pt_net
    model.load_state_dict(torch.load('./vd16_data_wpca.pkl'))
    if use_cuda:
        model.cuda()
    return model

def generate_list(datapath,batch_size):
    model = trans_main()
    descs = []
    is_gray = False
    jpeg_paths = sorted(glob.glob(os.path.join(datapath, '*.png')))
    
    for batch_offset in range(0, len(jpeg_paths), batch_size):
    # for batch_offset in range(0, 4, batch_size):
        images = []
        print(batch_offset)
        for i in range(batch_offset, batch_offset + batch_size):
            if i == len(jpeg_paths):
                break
            if is_gray:
                image = cv2.imread(jpeg_paths[i], cv2.IMREAD_GRAYSCALE)
                images.append(np.expand_dims(
                        np.expand_dims(image, axis=0), axis=0))
            else:
                image = cv2.imread(jpeg_paths[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(np.expand_dims(image, axis=0).transpose((0,3,1,2)))
        batch = np.concatenate(images, 0)
        im = torch.Tensor(batch)

        output = model(im.cuda())
        descs =  descs + output.tolist()
        tmp = 1

        # descs = descs + list(self.sess.run(
        #         self.net_out, feed_dict={self.tf_batch: batch}))
    
    tmp = 1
    return descs

def get_matrix(feats):
    use_dim = 128
    use_feats = np.array(feats)[:, :use_dim]
    output = open('./feats_00.pkl', 'wb')
    pickle.dump(feats, output)
    output.close()
    score = np.dot(np.array(feats),np.array(feats).T)
    sns.heatmap(score,annot=False,cmap='RdYlGn') #data.corr()-->correlation matrix
    fig=plt.gcf()
    fig.set_size_inches(12,12)
    #plt.savefig('euclidean_distance_'+LAYER_NAME)
    plt.show()
    output = open('./feats_score_00.pkl', 'wb')
    pickle.dump(score, output)
    output.close()
    #%% Suppressing a certain radius around the diagonal to prevent self-matches
    # (not real loop closures).
    suppression_diameter = 501
    diag_suppression = scisig.convolve2d(
            np.eye(use_feats.shape[0]), np.ones((1, suppression_diameter)), 'same')

    #%% NetVLAD matching, reject along diagonal (query ~ match):
    sq_dists = scid.squareform(scid.pdist(use_feats, 'sqeuclidean'))
    sq_dists[diag_suppression > 0] = np.inf
    plt.imshow(sq_dists)
    plt.title('Confusion matrix NetVLAD')
    plt.colorbar()
    plt.show()
    
if __name__ == "__main__":
    feats = generate_list('image_2',2)
    
    get_matrix(feats)

