import torch


class DistanceTransformLoss(torch.nn.Module):
    def __init__(self, border_factor=0.2, ignore_index=255, reduction='sum', weight=None):
        super(DistanceTransformLoss, self).__init__()
        self.border_factor = border_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight
        
        self.target_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction, 
            weight=weight
        )

    def get_class_boundaries(self, matrix):
        border_tb  = matrix[:,:,1:,:] - matrix[:,:,:-1,:] 
        border_lr  = matrix[:,:,:,1:] - matrix[:,:,:,:-1]
        border_tb  = torch.nn.functional.pad(input=border_tb, pad=[0,0,0,1,0,0], mode='constant', value=0) 
        border_lr  = torch.nn.functional.pad(input=border_lr, pad=[0,1,0,0,0,0], mode='constant', value=0)
        border_onehot = (border_tb + border_lr) != 0
        return border_onehot

    def get_distance_matrix(self, matrix, dist_transform=True):
        bound = torch.tensor(matrix.shape).sum()
        dist_matrix = torch.ones_like(matrix).float() * bound
        pos_indices = torch.nonzero(matrix == 1, as_tuple=True)
        dist_matrix[pos_indices] = 0
        if dist_transform:
            # Update distance transform matrix by repeating max_pool2d min(H,W) times [BCHW]
            for _ in range(min(matrix.shape[2], matrix.shape[3])): 
                maxp_matrix = -torch.nn.functional.max_pool2d(input=-dist_matrix, kernel_size=3, padding=1, stride=1)
                dist_matrix = torch.min(dist_matrix, maxp_matrix+1)
        return dist_matrix

    def border_criterion(self, slices, targets):
        slices_max = torch.argmax(slices, dim=1, keepdim=True)
        slices_borders = self.get_class_boundaries(slices_max)
        target_borders = self.get_class_boundaries(targets)
        target_borders = self.get_distance_matrix(target_borders, dist_transform=True)
        border_penalty = slices_borders * target_borders
        if   self.reduction == 'mean': return border_penalty[border_penalty != 0].mean()
        elif self.reduction ==  'sum': return border_penalty[border_penalty != 0].sum()
        else:                          return border_penalty[border_penalty != 0]
         
    def forward(self, slices, targets):
        target_loss = self.target_criterion(slices, torch.squeeze(targets, dim=1))
        border_loss = self.border_criterion(slices, targets)
        if (self.reduction is None) or (self.reduction == 'none'):
            return (target_loss, border_loss)
        return target_loss + (self.border_factor * border_loss)
