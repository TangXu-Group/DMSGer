from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math
from torch.nn import init
class Graph2dConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        block_num,
        adj_mask = None
    ):
        super(Graph2dConvolution, self).__init__()
        
        self.weight = Parameter(torch.Tensor(in_channels, out_channels, 1, 1))
        self.W = Parameter(torch.Tensor(out_channels, out_channels, 1, 1))
        
        self.reset_parameters()
        
        self.in_features = in_channels
        self.out_features = out_channels
        self.block_num = block_num
        self.adj_mask = adj_mask
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
    def forward(self, input, index):
        input = (input.permute(0,2,3,1)).matmul(self.weight[:,:,0,0]).permute(0,3,1,2)
        
        index = nn.UpsamplingNearest2d(size = (input.shape[2],input.shape[3]))(index.float()).long()
        block_within_index =  list(set(np.array(seg_index.view(-1))))
        
        batch_size = input.shape[0]
        channels = input.shape[1]

        # computing the regional mean of input
        input_means = []
        for i in range(len(block_within_index)):
            block_mask = (index == block_within_index[i]).float()
            sum_block = torch.sum(block_mask,dim = (2,3))
            sum_input = torch.sum(input * block_mask, dim = (2,3))
            mean_input = sum_input/sum_block
            input_means.append(mean_input)
        input_means = torch.stack(input_means).permute(1,0,2)
        
        # computing the adjance metrix
        input_means_ = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
        input_means_ = (input_means_ - input_means.unsqueeze(1)).permute(0, 2, 1, 3)
        M = (self.W[:,:,0,0]).mm(self.W[:,:,0,0].T)
        adj = input_means_.reshape(batch_size, -1, channels).matmul(M)
        adj = torch.sum(adj * input_means_.reshape(batch_size, -1, channels),dim=2).view(batch_size, self.block_num,self.block_num)
        adj = torch.exp(-1 * adj)+ torch.eye(self.block_num).repeat(batch_size, 1, 1).cuda()
        if self.adj_mask is not None:
            adj = adj * self.adj_mask
        
        # generating the adj_mean
        adj_means = input_means.repeat(self.block_num,1,1,1).permute(1,0,2,3) * adj.unsqueeze(3)
        adj_means = (1-torch.eye(self.block_num).reshape(1,self.block_num,self.block_num,1).cuda()) * adj_means
        adj_means = torch.sum(adj_means, dim=2) # batch_size，self.block_num, channel_num
        
        #obtaining the graph update features
        for i in range(len(block_within_index)):
            block_mask = (index == block_within_index[i]).float()
            update_input = input * block_mask + adj_means[:,i].unsqueeze(2).unsqueeze(3)
            input = input * (1-block_mask) + update_input
        return input

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'