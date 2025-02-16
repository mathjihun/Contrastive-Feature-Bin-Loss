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

    _gt_bin_indices = torch.zeros([len(_gt)], dtype=torch.long).cuda()

    for threshold in centers:
        _gt_bin_indices[_gt>threshold] += 1

    """
    diverse part
    """
    u_value, u_index, u_counts = torch.unique(_gt_bin_indices, return_inverse=True, return_counts =True)
    center_f = torch.zeros([len(u_value), f_c]).cuda()
    center_f.index_add_(0, u_index, _features)
    u_counts = u_counts.unsqueeze(1)
    center_f = center_f / u_counts

    p = F.normalize(center_f, dim=1)     # center_f는 target 개수만큼 행이 존재한다. 이를 평균을 계산하고 normalize까지 해주었다.
    _distance = euclidean_dist(p, p)

    if _distance.size(0) > 1:           # gt가 동일한 label로만 구성되어 있을 경우 _distance가 없는 문제가 발생함
    # 참고로 이런 경우에도 밑의 _tightness는 계산될 수 있다. 
    # 예를 들어 모두 10인 label로 구성되어 있더라도 feature가 다 다를 것이기에 center를 잘 구할 수 있고 이를 통하여 loss 계산을 할 수 있다.
        _distance = up_triu(_distance)
        
        #c_u_value = centers[u_value].unsqueeze(dim=1)

        #_weight = euclidean_dist(c_u_value, c_u_value)
        #_weight = up_triu(_weight)

        #_max = torch.max(_weight)
        #_min = torch.min(_weight)
        #_weight = ((_weight - _min) / _max)

        #_distance = _distance * _weight

        _entropy = torch.mean(_distance)
        loss = loss - _entropy

    """
    tightness part
    """
    _features = F.normalize(_features, dim=1)
    _features_center = p[u_index, :]        # 모든 feature들의 개수가 847 개라고 하면 channel 128일 때 847 x 128의 텐서가 나온다. 전부 각각에 해당하는 center로 되어 있다.
    _features = _features - _features_center
    _features = _features.pow(2)
    _tightness = torch.sum(_features, dim=1)
    _mask = _tightness > 0
    _tightness = _tightness[_mask]

    if _tightness.size(0) != 0:
        # _tightness = torch.sqrt(_tightness)
        _tightness = torch.mean(_tightness)
        loss = loss + _tightness

    return loss
 


def cfbloss(features, gt, mean_centers, dataset='nyu', mask=None):
    """
    Features: a certain layer's features
    gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
    mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
    """

    if dataset=='nyudepthv2' or dataset=='nyu':
        cut = 0.001
    else:
        cut = 1.0

    # interpolation 모두 제거함
    f_n, f_c, f_h, f_w = features.size()   

    features = features.permute(0, 2, 3, 1)  # n, h, w, c
    features = torch.flatten(features, start_dim=1, end_dim=2)
    #features2 = features2.permute(0, 2, 3, 1)  # n, h, w, c
    #features2 = torch.flatten(features2, start_dim=1, end_dim=2)

    gt = F.interpolate(gt, size=[f_h, f_w], mode='nearest')
    #gt2 = F.interpolate(gt, size=[f_h, f_w], mode='nearest')

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

        '''
        concat_features = torch.cat([_features, _features2], dim=0)
        concat_gt = torch.cat([_gt, _gt2], dim=0)

        batchwise_loss = _diverse_tightness(_features=concat_features, _gt=concat_gt, f_c=f_c)
        '''

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
