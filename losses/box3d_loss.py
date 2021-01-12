import torch
import torch.nn as nn



class Box3DLoss(nn.Module):
    """"""

    def __init__(self, cfg, alpha=2, beta=4, w_off=1, w_height=0.1, w_sizes=0.03, w_angle=0.2, w_head=0.1, w_heat=0.0008):
        super().__init__()
        self.l1 = nn.SmoothL1Loss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.beta = beta
        self.w_off = w_off
        self.w_height = w_height
        self.w_sizes = w_sizes
        self.w_angle = w_angle
        self.w_head = w_head
        self.w_heat = w_heat


    def forward(self, x, y):

        offsets_pred, height_pred, sizes_pred, angle_pred, heading_pred, heatmap_pred = self.parse_array(x)
        offsets_true, height_true, sizes_true, angle_true, heading_true, heatmap_true = self.parse_array(y[0])

        offsets_pred = torch.sigmoid(offsets_pred)
        sizes_pred = torch.relu(sizes_pred)
        heatmap_pred = torch.clamp(torch.sigmoid(heatmap_pred), 1e-4, 1-1e-4)

        mask_heat = torch.floor(heatmap_true)
        mask_obj = torch.clamp(torch.sum(mask_heat, dim=1), 0, 1).unsqueeze(1)
        N = torch.sum(mask_obj)
        N_cells = heatmap_true[0,0].shape[0]*heatmap_true[0,0].shape[1]

        offsets_loss = self.w_off * torch.sum(mask_obj.expand_as(offsets_pred) * self.l1(offsets_pred, offsets_true)) / (N + 1e-4)
        height_loss = self.w_height * torch.sum(mask_obj.expand_as(height_pred) * self.l1(height_pred, height_true)) / (N + 1e-4)
        sizes_loss = self.w_sizes * torch.sum(mask_obj.expand_as(sizes_pred) * self.l1(sizes_pred, sizes_true)) / (N + 1e-4)
        angle_loss = self.w_angle * torch.sum(mask_obj.expand_as(angle_pred) * self.l1(angle_pred, angle_true)) / (N + 1e-4)
        heading_loss = self.w_head * torch.sum(mask_obj.expand_as(heading_pred) * self.bce_logits(heading_pred, heading_true)) / (N + 1e-4)

        heatmap_loss_pos = -self.w_heat * torch.sum(mask_heat * torch.pow((1-heatmap_pred), self.alpha) * torch.log(heatmap_pred)) / N_cells
        heatmap_loss_neg = -self.w_heat * torch.sum((1-mask_heat) * torch.pow((1-heatmap_true), self.beta) * torch.pow(heatmap_pred, self.alpha) * torch.log(1-heatmap_pred)) / N_cells
        heatmap_loss = heatmap_loss_neg if N == 0 else (heatmap_loss_pos + heatmap_loss_neg) / N

        if N == 0:
            return heatmap_loss

        return (offsets_loss + height_loss + sizes_loss + angle_loss + heading_loss + heatmap_loss)

    
    def parse_array(self, x):
        offsets = x[:,0:2]
        height = x[:,2:3]
        sizes = x[:,3:6]
        angle = x[:,6:7]
        heading = x[:,7:8]
        heatmap = x[:,8:]
        return (offsets, height, sizes, angle, heading, heatmap)
