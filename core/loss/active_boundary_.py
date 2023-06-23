import torch


class ActiveBoundaryLoss(torch.nn.Module):
    """
    Active Boundary Loss for Semantic Segmentation 
    Paper:  https://ojs.aaai.org/index.php/AAAI/article/view/20139
    GitHub: https://github.com/wangchi95/active-boundary-loss
    ****** RAFAEL VARETO'S ORIGINAL IMPLEMENTATION ****** 
    """
    def __init__(self, device='cpu', ignore_index=255, reduction='sum', smoothing=0.0, upperbound=20., weight=None):
        super(ActiveBoundaryLoss, self).__init__()
        self.device        = device
        self.ignore_index  = ignore_index
        self.reduction     = reduction
        self.smoothing     = smoothing
        self.upperbound    = upperbound

        self.border_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=smoothing
        )
        self.target_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction, 
            weight=weight
        )

    @staticmethod
    def kl_divergence(q, p):
        """Kullback-Leibler Divergence"""
        return torch.nn.functional.softmax(p, dim=1) * (torch.nn.functional.log_softmax(p, dim=1) - torch.nn.functional.log_softmax(q, dim=1)) 

    @staticmethod
    def radius_weight(reduction, distance, upperbound):
        radial_weight = torch.clamp(distance, max=upperbound) / upperbound
        if   reduction == 'mean': return radial_weight.mean()
        elif reduction == 'sum':  return radial_weight.sum()
        return radial_weight

    def logits2boundary(self, logits, edge_ratio=0.05, eps=1e-5):
        _, _, h, w  = logits.shape
        pixel_ratio = (h * w) * edge_ratio
        # Compute Kullback-Leibler Divergence in top-bottom and left-right fashion
        kl_tb  = ActiveBoundaryLoss.kl_divergence(logits[:,:,1:,:], logits[:,:,:-1,:]).sum(dim=1, keepdim=True)
        kl_lr  = ActiveBoundaryLoss.kl_divergence(logits[:,:,:,1:], logits[:,:,:,:-1]).sum(dim=1, keepdim=True)
        kl_tb  = torch.nn.functional.pad(input=kl_tb, pad=[0,0,0,1,0,0,0,0], mode='constant', value=0) # pad=[left/right/top/bottom/front/back,...]
        kl_lr  = torch.nn.functional.pad(input=kl_lr, pad=[0,1,0,0,0,0,0,0], mode='constant', value=0) # pad=[left/right/top/bottom/front/back,...]
        kl_sum = kl_lr + kl_tb
        # Encounter the adaptive eps threshold
        kl_bin = (kl_sum > eps)
        while kl_bin.sum() > pixel_ratio:
            eps = eps * 1.2
            kl_bin = (kl_sum > eps)
        # Dilate borders with convolution
        dilate_kernel = torch.ones((1,1,3,3)).to(self.device)
        dilation = torch.nn.functional.conv2d(kl_bin.float(), dilate_kernel, stride=1, padding=1)
        return dilation > 0

    def gtruth2boundary(self, gtruth, ignore_index=-1):
        gt_tb  = gtruth[:,:,1:,:] - gtruth[:,:,:-1,:]  # BCHW
        gt_lr  = gtruth[:,:,:,1:] - gtruth[:,:,:,:-1]
        gt_tb  = torch.nn.functional.pad(input=gt_tb, pad=[0,0,0,1,0,0], mode='constant', value=0) != 0 
        gt_lr  = torch.nn.functional.pad(input=gt_lr, pad=[0,1,0,0,0,0], mode='constant', value=0) != 0
        gt_sum = gt_lr + gt_tb
        # set 'ignore area' to all boundary
        gt_sum += (gtruth==ignore_index)
        return gt_sum > 0

    def get_orientation(self, target_dist, slices_edge, slices, eps=1e-5, max_dis=1e5):
        slices_index = torch.nonzero(slices_edge, as_tuple=False)
        n,c,x,y = slices_index.T
        # Permuting BCHW->BHWC and padding parameters in form N(H+2)(W+2)C 
        slices = slices.permute(0,2,3,1)
        slices_p = torch.nn.functional.pad(slices.detach(), (0,0,1,1,1,1), mode='replicate') # pad=[left/right/top/bottom/front/back,...]
        target_dist_p = torch.nn.functional.pad(target_dist,(1,1,1,1,0,0), mode='constant', value=max_dis)
        """
        Defining pixel neghborhood:
            | 4 | 0 | 5 |
            | 2 | 8 | 3 |
            | 6 | 1 | 7 |
        """
        neighbor_row = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        neighbor_col = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        kl_div_matrix = torch.zeros((0, len(x))).to(self.device)
        radius_matrix = torch.zeros((0, len(x))).to(self.device)
        # compare central with neighboring logits 
        center_logits = slices[(n,x,y)]
        for nx, ny in zip(neighbor_row, neighbor_col):
            radius_record = target_dist_p[(n, c, x+nx+1, y+ny+1)]
            radius_matrix = torch.cat((radius_matrix, radius_record.unsqueeze(0)), 0)
            if (nx != 0) or (ny != 0):
                nearby_logits = slices_p[(n, x+nx+1, y+ny+1)]
                kl_div_record = ActiveBoundaryLoss.kl_divergence(center_logits, nearby_logits).sum(1)
                kl_div_matrix = torch.cat((kl_div_matrix, kl_div_record.unsqueeze(0)), 0)
        # obtaining orientation index, transposing
        radius_matrix = torch.argmin(radius_matrix, dim=0)
        kl_div_matrix = torch.transpose(kl_div_matrix, 0, 1)
        direction_idx = [radius_matrix != 8]
        # skipping local minimum (position=8)
        target_track = radius_matrix[direction_idx]
        slices_track = kl_div_matrix[direction_idx]
        weight_track = target_dist[(n,c,x,y)][direction_idx]
        return target_track, slices_track, weight_track

    def get_dist_matrix(self, boundaries, dist_transform=True):
        bound = torch.tensor(boundaries.shape, requires_grad=False).sum()
        dist_matrix = torch.ones_like(boundaries).float() * bound
        pos_indices = torch.nonzero(boundaries == 1, as_tuple=True)
        dist_matrix[pos_indices] = 0
        if dist_transform:
            # Update distance transform matrix by repeating max_pool2d min(H,W) times [BCHW]
            for _ in range(min(boundaries.shape[2], boundaries.shape[3])): 
                maxp_matrix = -torch.nn.functional.max_pool2d(input=-dist_matrix, kernel_size=3, padding=1, stride=1)
                dist_matrix = torch.min(dist_matrix, maxp_matrix+1)
        return dist_matrix.detach()

    def forward(self, slices, targets, target_distance=None):
        sb, sc, sh, sw = slices.shape
        tb, tc, th, tw = targets.shape
        assert (sb == tb) and (sc > tc) and (sh == th) and (sw == tw)
        # If target distance matrix is no given, compute it
        if target_distance is None:
            target_boundary = self.gtruth2boundary(targets, ignore_index=self.ignore_index)
            target_distance = self.get_dist_matrix(target_boundary)
        slices_boundary = self.logits2boundary(slices)
        target_track, slices_track, weight_track = self.get_orientation(target_distance, slices_boundary, slices)
        # Compute border and target-based losses
        border_loss = self.border_criterion(slices_track, target_track) + ActiveBoundaryLoss.radius_weight(self.reduction, weight_track, self.upperbound)
        target_loss = self.target_criterion(slices, torch.squeeze(targets, dim=1))
        if (self.reduction is None) or (self.reduction == 'none'):
            return (target_loss, border_loss)
        return target_loss + border_loss


if __name__ == '__main__':
    from torch.backends import cudnn
    import os
    import random
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = 'cpu'

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    b,c,h,w = 12,1,12,12
    slices = torch.randn((b,c*6,h,w)).to(device)
    gtruth = torch.zeros((b,c,h,w)).long().to(device)
    for h_ in range(h):
        slices[h_,h_//2,h_,:] = 1
        # gtruth[h_,:,h_,:] = h_//2
        for idx in range(h_):
            for jdx in range(h_):
                gtruth[h_,:,idx,jdx] = h_//2
    print(slices.shape, gtruth.shape)

    abl = ActiveBoundaryLoss(reduction='mean')
    loss = abl(slices, gtruth)
    print(loss)
