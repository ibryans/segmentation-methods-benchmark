import torch


class RunningScore(object):
    def __init__(self, n_classes, threshold=None):
        self.bfs_threshold = threshold
        self.bfs_matrix = [0.] if self.bfs_threshold is None else list()
        self.confusion_matrix = torch.zeros((n_classes, n_classes))
        self.epoch_counter = 0
        self.n_classes = n_classes

    def __compute_matrix(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = torch.bincount(n_class * label_true[mask].to(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist.cpu()

    def __find_class_contours(self, matrix, label):
        lbl_matrix = (matrix == label).int()
        gt_tb  = lbl_matrix[1:,:] - lbl_matrix[:-1,:]
        gt_lr  = lbl_matrix[:,1:] - lbl_matrix[:,:-1]
        gt_tb  = torch.nn.functional.pad(input=gt_tb, pad=[0,0,0,1], mode='constant', value=0) != 0 
        gt_lr  = torch.nn.functional.pad(input=gt_lr, pad=[0,1,0,0], mode='constant', value=0) != 0
        gt_idx = torch.nonzero((gt_lr + gt_tb) == 1, as_tuple=False)
        return gt_idx

    def __compute_boundary(self, label_trues, label_preds):
        device_idx = label_trues.get_device()
        device = device_idx if device_idx >= 0 else 'cpu'
        bf_scores = torch.zeros(self.n_classes, device=device)
        for label in range(self.n_classes):
            # Removing len=1 axes from matrices
            preds, trues = label_preds.squeeze(), label_trues.squeeze()
            # Find matrix indices storing boundaries
            contour_pr = self.__find_class_contours(preds, label)
            contour_tr = self.__find_class_contours(trues, label)
            # Compute BF1 Score
            if len(contour_pr) and len(contour_tr):
                # Compute Precision and Recall
                precis, pre_num, pre_den = self.__precision_recall(contour_tr, contour_pr, self.bfs_threshold)
                recall, rec_num, rec_den = self.__precision_recall(contour_pr, contour_tr, self.bfs_threshold)
                bf_scores[label] += (2 * recall * precis / (recall + precis)) if (recall + precis) > 0 else 0.
            else: bf_scores[label] += 0
        return bf_scores
    
    def __precision_recall(self, vector_a, vector_b, threshold=2):
        """ 
        For precision, vector_a==GT & vector_b==Prediction
        For recall, vector_a==Prediction & vector_b==GT 
        """
        # Constrain long arrays when their size differ significantly
        upper_bound = max([len(vector_a), len(vector_b)])
        lower_bound = min([len(vector_a), len(vector_b)])
        bound = upper_bound if (upper_bound/lower_bound <= 2.) else lower_bound
        # Shrinking vectors 
        vector_a = vector_a[:bound].float()
        vector_b = vector_b[:bound].float()
        # Efficient and ellegant implementation of Euclidean Distance
        distance  = torch.cdist(vector_a, vector_b, p=2)
        top_count = torch.any(distance < threshold, dim=0).sum()
        # Not very ellegant implementation of Euclidean Distance (time consuming)
        # hits = list()
        # for point_b in vector_b:
        #     distance = torch.square(vector_a[:,0] - point_b[0]) + torch.square(vector_a[:,1] - point_b[1])
        #     hits.append(torch.any(distance < threshold**2))
        # top_count = torch.sum(torch.as_tensor(hits))
        try:
            precision_recall = top_count / len(vector_b)
        except ZeroDivisionError:
            precision_recall = 0
        return precision_recall, top_count, len(vector_b)

    def update(self, slices, targets):
        label_preds, label_trues = slices.detach().max(dim=1)[1], targets.detach() #.cpu().numpy()
        for lp, lt in zip(label_preds, label_trues):
            self.confusion_matrix += self.__compute_matrix(lt.flatten(), lp.flatten(), self.n_classes)
            if self.bfs_threshold is not None:
                self.bfs_matrix.append(self.__compute_boundary(lt, lp)) 

    def get_scores(self, epoch=None):
        """
        Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        self.epoch_counter = (self.epoch_counter + 1) if epoch is None else epoch
        bfs = torch.mean(torch.stack(self.bfs_matrix))
        hist = self.confusion_matrix
        acc = torch.diag(hist).sum() / hist.sum()
        acc_cls = torch.diag(hist) / hist.sum(dim=1)
        mean_acc_cls = torch.nanmean(acc_cls)
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        mean_iu = torch.nanmean(iu)
        freq = hist.sum(dim=1) / hist.sum() # fraction of the pixels that come from each class
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        return {'Epoch': self.epoch_counter,
                'Decision': mean_iu.item() if self.bfs_threshold is None else bfs.item(),
                'BF1 Score': bfs.item(),
                'Pixel Acc': acc.item(),
                'Class Accuracy': acc_cls.tolist(),
                'Mean Class Acc': mean_acc_cls.item(),
                'Freq Weighted IoU': fwavacc.item(),
                'Mean IoU': mean_iu.item(),
                'Confusion Matrix': self.confusion_matrix.tolist()}

    def reset(self):
        self.bfs_matrix = [0] if self.bfs_threshold is None else list()
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))
        

if __name__ == '__main__':
    array_a = torch.random.randint(50, size=(1000, 2))
    array_b = torch.random.randint(50, size=(1000, 2))
    bf_precision_recall(array_a, array_b, threshold=2)
