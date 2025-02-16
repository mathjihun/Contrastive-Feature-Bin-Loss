"""

"""
import torch
import torch.nn.functional as F

"""

"""
import torch
import torch.nn.functional as F

def _diverse_tightness(_features, _gt, f_c, centers):
    loss = 0

    even_mask = _gt <= centers[-2]         # 9.7보다 큰 값 버림
    odd_mask = _gt > centers[1]            # 0.7이하 버림

    _gt_bin_indices_even = torch.zeros([len(_gt[even_mask])], dtype=torch.long).cuda()     # 9.7보다 큰 값 버림
    _gt_bin_indices_odd = torch.zeros([len(_gt[odd_mask])], dtype=torch.long).cuda()       # 0.7이하 버림


    for threshold in centers[2::2]:                 # 0~1.1, 1.1~1.5, ..., 9.4~9.7, (9.7~10.0)임
        _gt_bin_indices_even[_gt[even_mask]>threshold] += 1 
            

    for threshold in centers[3::2]:                    # (0~0.7), 0.7~1.3, 1.3~1.6, ..., 9.3~9.5, 9.5~10.0임
        _gt_bin_indices_odd[_gt[odd_mask]>threshold] += 1

    # 위의 약간의 문제는 (9.7~10.0)과 (0~0.7)이다. 다른 것들은 두 개의 bin을 사용하는데 이 둘은 하나만 사용함 필요가 없음

    # 이때 u_index_even과 u_index_odd는 각각 0~127의 성분 (128)을 갖게 된다. 다시 말해 (centers.size(0)-2) / 2
    u_value_even, u_index_even, u_counts_even = torch.unique(_gt_bin_indices_even, return_inverse=True, return_counts=True)
    u_value_odd, u_index_odd, u_counts_odd = torch.unique(_gt_bin_indices_odd, return_inverse=True, return_counts=True)

    center_f_even = torch.zeros([len(u_value_even), f_c]).cuda()
    center_f_odd = torch.zeros([len(u_value_odd), f_c]).cuda()

    center_f_even.index_add_(0, u_index_even, _features[even_mask, :])   # 9.7보다 큰 값 버림
    center_f_odd.index_add_(0, u_index_odd, _features[odd_mask, :])      # 0.7이하 버림

    u_counts_even = u_counts_even.unsqueeze(1)
    u_counts_odd = u_counts_odd.unsqueeze(1)
        
    center_f_even = center_f_even / u_counts_even
    center_f_odd = center_f_odd / u_counts_odd

    p_even = F.normalize(center_f_even, dim=1)
    p_odd = F.normalize(center_f_odd, dim=1)

    p = torch.cat([p_even, p_odd], dim=0)

    _distance = euclidean_dist(p, p)

    if _distance.size(0) > 1:               # 위에서 1 x 1 행렬나올 경우 다시 말해 모두가 하나의 bin에 속할 때 여기서 문제 생김 (밀어낼 게 없으므로)
        _distance = up_triu(_distance)
        

        #c_u_value = torch.cat([centers[2::2][u_value_even], centers[3::2][u_value_odd]], dim=0).unsqueeze(1)

        #_weight = euclidean_dist(c_u_value, c_u_value)
        #_weight = up_triu(_weight)
        #_max = torch.max(_weight)
        #_min = torch.min(_weight)
        #_weight = ((_weight - _min) / _max)
        #_weight = _weight / centers[-1]

        #_distance = _distance * _weight 

        _entropy = torch.mean(_distance)

        loss = loss - _entropy
    

    """
    tightness part
    """
    _features = F.normalize(_features, dim=1)

    _features_even = (_features[even_mask, :] - p_even[u_index_even, :]).pow(2)
    _features_odd = (_features[odd_mask, :] - p_odd[u_index_odd, :]).pow(2)

    _tightness = torch.cat([torch.sum(_features_even, dim=1), torch.sum(_features_odd, dim=1)], dim=0)
    _mask = _tightness > 0

    _tightness = _tightness[_mask]

    if _tightness.size(0) != 0:
        #_tightness = torch.sqrt(_tightness)
        _tightness = torch.mean(_tightness)
        loss = loss + _tightness


    return loss
 


def cfbloss(features, gt, mean_centers, dataset='nyu', mask=None):
    """
    Features: a certain layer's features
    gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
    mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
    """

    if dataset=='nyu' or 'nyudepthv2':
        cut = 0.001
    else:
        cut = 1.0

    
    # interpolation 모두 제거함
    f_n, f_c, f_h, f_w = features.size()   

    '''     
    # q4 style
    features = features.permute(0, 2, 3, 1)  # n, h, w, c
    features = torch.flatten(features, start_dim=1, end_dim=2)
    #features2 = features2.permute(0, 2, 3, 1)  # n, h, w, c
    #features2 = torch.flatten(features2, start_dim=1, end_dim=2)

    gt = F.interpolate(gt, size=[f_h, f_w], mode='nearest')
    #gt2 = F.interpolate(gt2, size=[f_h, f_w], mode='nearest')
    '''
    
     
    # q0 style
    features = F.interpolate(features, size=[f_h//4, f_w//4], mode='nearest')
    features = features.permute(0,2,3,1)
    features = torch.flatten(features, start_dim=1, end_dim=2)

    #features2 = F.interpolate(features2, size=[f_h//4, f_w//4], mode='nearest')
    #features2 = features2.permute(0,2,3,1)
    #features2 = torch.flatten(features2, start_dim=1, end_dim=2)

    gt = F.interpolate(gt, size=[f_h//4, f_w//4], mode='nearest')
    #gt2 = F.interpolate(gt2, size=[f_h//4, f_w//4], mode='nearest')
     

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
        
        '''
        _gt2 = gt2[i,:].view(-1)
        _mask2 = _gt2 > cut
        _mask2 = _mask2.to(torch.bool)
        _gt2 = _gt2[_mask2]
        _features2 = features2[i,:]
        _features2 = _features2[_mask2,:]
        '''

        _mean_centers = mean_centers[i,:]
        centers = _mean_centers.squeeze()
       
        #concat_features = torch.cat([_features, _features2], dim=0)
        #concat_gt = torch.cat([_gt, _gt2], dim=0)

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
