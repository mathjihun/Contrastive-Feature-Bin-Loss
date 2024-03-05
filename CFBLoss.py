import torch
import torch.nn.functional as F

def _diverse_tightness(_features, _gt, f_c, centers):
    '''
    centers = [a_1, a_2, a_3, ..., a_{2k-1}, a_{2k}]

    even_mask 
    [a_1, a_3]: 0, [a_3, a_5]: 1, ... , [a_{2k-3}, a_{2k-1}]: k-2
    
    odd_mask
    [a_2, a_4]: 0, [a_4, a_6]: 1, ... , [a_{2k-2}, a_{2k}]: k-2

    '''
    loss = 0

    even_mask = _gt <= centers[-2]         # delete value higher than a_{2k-1}
    odd_mask = _gt > centers[1]            # delete value lower than a_2

    _gt_bin_indices_even = torch.zeros([len(_gt[even_mask])], dtype=torch.long).cuda()
    _gt_bin_indices_odd = torch.zeros([len(_gt[odd_mask])], dtype=torch.long).cuda()

    # indexing
    for threshold in centers[2::2]:                 
        _gt_bin_indices_even[_gt[even_mask]>threshold] += 1 
            
    # indexing
    for threshold in centers[3::2]:                 
        _gt_bin_indices_odd[_gt[odd_mask]>threshold] += 1

    # u_index_even & u_index_odd have values 0 to (centers.size(0)-2)/2-1
    u_value_even, u_index_even, u_counts_even = torch.unique(_gt_bin_indices_even, return_inverse=True, return_counts=True)
    u_value_odd, u_index_odd, u_counts_odd = torch.unique(_gt_bin_indices_odd, return_inverse=True, return_counts=True)

    center_f_even = torch.zeros([len(u_value_even), f_c]).cuda()
    center_f_odd = torch.zeros([len(u_value_odd), f_c]).cuda()

    center_f_even.index_add_(0, u_index_even, _features[even_mask, :]) 
    center_f_odd.index_add_(0, u_index_odd, _features[odd_mask, :])   

    u_counts_even = u_counts_even.unsqueeze(1)
    u_counts_odd = u_counts_odd.unsqueeze(1)
        
    center_f_even = center_f_even / u_counts_even     # mean feature vector of even-numbered bins
    center_f_odd = center_f_odd / u_counts_odd        # mean feature vector of odd-numbered bins

    p_even = F.normalize(center_f_even, dim=1)
    p_odd = F.normalize(center_f_odd, dim=1)

    p = torch.cat([p_even, p_odd], dim=0)             # mean feature vector of overlapping bins

    _distance = euclidean_dist(p, p)                  # calculate diverse loss

    if _distance.size(0) > 1:               # if _distance is 1 x 1 matrix, then diverse loss is zero
        _distance = up_triu(_distance)          
        _entropy = torch.mean(_distance)

        loss = loss - _entropy
    

    """
    tightness part
    """
    _features = F.normalize(_features, dim=1)

    # || all valid pixels in each bin - mean feature vector of bin ||^2
    _features_even = (_features[even_mask, :] - p_even[u_index_even, :]).pow(2)
    _features_odd = (_features[odd_mask, :] - p_odd[u_index_odd, :]).pow(2)

    _tightness = torch.cat([torch.sum(_features_even, dim=1), torch.sum(_features_odd, dim=1)], dim=0)
    _mask = _tightness > 0

    _tightness = _tightness[_mask]

    if _tightness.size(0) != 0:             # if all bins have only one value, then tightness loss is zero
        #_tightness = torch.sqrt(_tightness)
        _tightness = torch.mean(_tightness)
        loss = loss + _tightness


    return loss
 


def cfbloss(features, gt, mean_centers, dataset='nyudepthv2', mask=None):
    """
    Features: a certain layer's features
    gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
    mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
    """

    if dataset=='nyudepthv2' or dataset=='nyu':
        cut = 0.001
    else:
        cut = 1.0

    f_n, f_c, f_h, f_w = features.size()
    
    # encoder style
    features = features.permute(0, 2, 3, 1)  # batch x h x w x C
    features = torch.flatten(features, start_dim=1, end_dim=2)   # batch x S x C

    gt = F.interpolate(gt, size=[f_h, f_w], mode='nearest')  # batch x 1 x h x w
    
    '''
    # decoder style
    features = F.interpolate(features, size=[f_h//4, f_w//4], mode='nearest')
    features = features.permute(0,2,3,1)
    features = torch.flatten(features, start_dim=1, end_dim=2)

    gt = F.interpolate(gt, size=[f_h//4, f_w//4], mode='nearest')
    ''' 

    loss = 0

    for i in range(f_n):
        """
        mask pixels that without valid values
        """
        _gt = gt[i,:].view(-1)
        _mask = _gt > cut
        _mask = _mask.to(torch.bool)
        _gt = _gt[_mask]
        _features = features[i,:]
        _features = _features[_mask,:]

        _mean_centers = mean_centers[i,:]
        centers = _mean_centers.squeeze()

        batchwise_loss = _diverse_tightness(_features, _gt, f_c, centers)

        loss += batchwise_loss


    return loss/ f_n

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def up_triu(x):
    # return a flattened view of up triangular elements of a square matrix
    n, m = x.shape
    assert n == m
    _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
    return x[_tmp]
