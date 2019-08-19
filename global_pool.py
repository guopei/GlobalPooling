'''
An implementation of common global pooling functions.
Borrowed some code from Cornnor Anderson's pooling.py
By Pei Guo
'''

import sys
import math
import torch
import warnings

class GlobalStochasticPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = sys.float_info.epsilon

    def forward(self, x):
        data = x.detach()
        b, c, h, w = data.size()
        data = data.view(-1, h*w)
        prob = torch.zeros_like(data)
        # returns one sampled index from last dimension
        # of data according to multinomial distribution.
        if self.training:
            # self.eps: avoid div by 0 in testing branch
            # and multinomial function
            indices = (data+self.eps).multinomial(1)
            # scatter_: set values at indices to 1
            prob.scatter_(1, indices, 1)
        # in case of tesing, use normalized prob instead
        # of sampling, data and x share memory, so don't do
        # data += self.eps.
        # # #
        # here we assume the input is non-negative TODO
        else:
            prob = data.div((data+self.eps).sum(-1, keepdim=True))
        # shape of y is b x c
        y = x.mul(prob.view(b,c,h,w)).sum(-1).sum(-1)
        return y

class GlobalLpNormPool2d(torch.nn.Module):
    def __init__(self, p = 2):
        super().__init__()
        self.p = p
        if self.p < 1:
            warning.warn("norm should be greater than or equal to 1," \
                    "automatically setting it to 1 to avoid errors.")
            self.p = 1 

    def forward(self, x):
        b, c, h, w = x.size()
        # equation from "a theoretical analysis of feature pooling ..."
        y = x.pow(self.p).mean(-1).mean(-1).pow(1 / self.p)
        return y

class GlobalLogAvgExpPool2d(torch.nn.Module):
    # this is essentially log-sum-exp function, 
    #
    def __init__(self, beta = 2):
        super().__init__()
        self.beta = beta
        if self.beta <= 0:
            warning.warn("norm should be greater than 0," \
                    "automatically setting it to 1e-3 to avoid errors.")
            self.beta = 1e-3

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(-1, h*w)
        # equation from "a theoretical analysis of feature pooling ..."
        # pytorch's implementation is numerically-stable I guess, see:
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/ 
        # for reference.
        y = 1 / self.beta * (x.logsumexp(-1).view(b,c) - math.log(h*w))
        return y

class GlobalSoftPool2d(torch.nn.Module):
    def __init__(self, beta = 1):
        super().__init__()
        self.beta = beta
        if self.beta < 0:
            warning.warn("beta should be greater than or equal to 0," \
                    "automatically setting it to 0 to avoid errors.")
            self.beta = 0 

    def forward(self, x):
        data = x.detach()
        b, c, h, w = data.size()
        data = data.view(b, c, h*w)
        prob = (data*self.beta).softmax(dim=-1)
        
        y = x.mul(prob.view(b,c,h,w)).sum(-1).sum(-1)
        return y

class GlobalKMaxPool2d(torch.nn.Module):
    def __init__(self, k = 1):
        super().__init__()
        self.k = k
        if self.k < 1:
            warning.warn("k should be greater than or equal to 1," \
                    "automatically setting it to 1 to avoid errors.")
            self.k = 1

    def forward(self, x):
        data = x.detach()
        b, c, h, w = data.size()
        if self.k >= h * w:
            warning.warn("k should be less than {} x {} = {}," \
                    "automatically setting it to {} to avoid errors.".format(
                        h, w, h*w, h*w-1))
            self.k = h*w - 1
        data = data.view(-1, h*w)
        # find self.k max indices
        _, indices = torch.topk(data, self.k)
        weight = torch.zeros_like(data)
        weight.scatter_(1, indices, 1)
        y = x.mul(weight.view(b,c,h,w)).sum(-1).sum(-1)
        return y

class GlobalMixedPool2d(torch.nn.Module):
    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = alpha
        if self.alpha > 1 or self.alpha < 0:
            warning.warn("alpha should be >= 0 and <= 1" \
                    "automatically setting it to 0 to avoid errors.")
            self.alpha = 0

    def forward(self, x):
        b, c, h, w = x.size()
        xmax = torch.nn.functional.adaptive_max_pool2d(x, 1)
        xavg = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        y = self.alpha * xmax + (1 - self.alpha) * xavg
        y = y.view(b, c)
        return y

class GlobalGatedPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = None

    def forward(self, x):
        data = x.detach()
        b, c, h, w = data.size()
        data = data.view(b * c, h * w)
        if self.weight is None:
            # Parameter is a special case of Tensor and
            # are good for paramter list using parameter()
            weight = torch.zeros((h*w,1), requires_grad=True)
            if x.is_cuda:
                weight = weight.cuda()
            self.weight = torch.nn.Parameter(weight)
            # initialization
            stdv = 1. / math.sqrt(self.weight.size(0))
            self.weight.data.uniform_(-stdv, stdv)

        xmax = torch.nn.functional.adaptive_max_pool2d(x, 1).view(b, c)
        xavg = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)

        alpha = torch.sigmoid(torch.matmul(data, self.weight)).view(b, c)
        y = alpha * xmax + (1 - alpha) * xavg
        y = y.view(b, c)
        return y

class GlobalDetailPreservePool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

# don't forget blur max, average, etc.
class GlobalBlurMaxPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

'''
# from github vadimkantorov
class CompactBilinearPool2d(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool = True):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: 
            torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]), 
            rand_s.float(), [input_dim, output_dim]).to_dense()
        self.sketch1 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim1,)), 
            2 * torch.randint(2, size = (input_dim1,)) - 1, input_dim1, output_dim), requires_grad = False)
        self.sketch2 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim2,)), 
            2 * torch.randint(2, size = (input_dim2,)) - 1, input_dim2, output_dim), requires_grad = False)

    def forward(self, x1, x2):
        fft1 = torch.rfft(x1.permute(0, 2, 3, 1).matmul(self.sketch1), signal_ndim = 1)
        fft2 = torch.rfft(x2.permute(0, 2, 3, 1).matmul(self.sketch2), signal_ndim = 1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        cbp = torch.irfft(fft_product, signal_ndim = 1, signal_sizes = (self.output_dim, )) * self.output_dim
        return cbp.sum(dim = [1, 2]) if self.sum_pool else cbp.permute(0, 3, 1, 2)
'''

def test():
    inputs = [torch.rand(2,3,4,5), torch.zeros(2,3,4,5)]
    pools = [GlobalStochasticPool2d(), GlobalLpNorm2d(), 
            GlobalSoftPool2d(), GlobalKMaxPool2d(), 
            GlobalLogAvgExpPool2d(), GlobalMixedPool2d(),
            GlobalGatedPool2d()]
    for x in inputs:
        for pool in pools:
            pool.train()
            y = pool(x)
            pool.eval()
            y = pool(x)
            assert(y.size() == (2,3))

    print("success! all test passed!")

if __name__ == "__main__":
    test()

