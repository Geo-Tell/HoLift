import torch
import torch.nn as nn
import torch.nn.functional as F

class L_cross(nn.Module):
    def __init__(self, cam_thre, temperature=0.1, conflict_thre=0.9):
        super(L_cross, self).__init__()
        self.cam_thre = cam_thre
        self.temperature = temperature
        self.conflict_thre = conflict_thre

    def forward(self, cam1, cam2, feat1, feat2, label1, label2):
        # norm cam
        B, C, H1, W1 = cam1.shape
        _, _, H2, W2 = cam2.shape
        D = feat1.shape[1]
        cam1 = F.relu(cam1) 
        cam2 = F.relu(cam2)
        cam1 = cam1 / (cam1.detach().amax((-1,-2),keepdim=True)+1e-8)
        cam2 = cam2 / (cam2.detach().amax((-1,-2),keepdim=True)+1e-8)

        cam1_flatten = cam1.reshape(B, C, -1)   # shape: [B, C, H*W]
        cam2_flatten = cam2.reshape(B, C, -1)
        feat1_flatten_t = feat1.reshape(B, D, -1).transpose(1,2)    # shape: [B, H*W, 512]
        feat2_flatten_t = feat2.reshape(B, D, -1).transpose(1,2)
        # exclude unlabelled class
        cam1_valid = cam1_flatten * label1.detach().unsqueeze(-1)
        cam2_valid = cam2_flatten * label2.detach().unsqueeze(-1)    # shape: [B, C, H*W]
        cam1_valid_cls = cam1_valid.argmax(1)   # shape: [B, H*W]
        cam2_valid_cls = cam2_valid.argmax(1)

        # verify mask class
        with torch.no_grad():
            one_hot = torch.eye(C).cuda()
            cam1_mask = one_hot.index_select(0, cam1_valid_cls.view(-1))
            cam2_mask = one_hot.index_select(0, cam2_valid_cls.view(-1))
            cam1_mask = cam1_mask.reshape(B, -1, C).permute(0,2,1)  # shape: [B, C, H*W]
            cam2_mask = cam2_mask.reshape(B, -1, C).permute(0,2,1)
            # exclude low confidence cam
            cam1_mask[cam1_valid<self.cam_thre] = torch.tensor(0.).cuda()
            cam2_mask[cam2_valid<self.cam_thre] = torch.tensor(0.).cuda()

            # exclude multi-class conflict cam
            conflict_mask1 = (cam1_valid > self.conflict_thre).sum(1) > 1   # shape: [B, H*W]
            conflict_mask2 = (cam2_valid > self.conflict_thre).sum(1) > 1
            cam1_mask_t = cam1_mask.permute(0,2,1)
            cam2_mask_t = cam2_mask.permute(0,2,1)
            cam1_mask_t[conflict_mask1] = torch.tensor(0.).cuda()
            cam2_mask_t[conflict_mask2] = torch.tensor(0.).cuda()
            cam1_mask = cam1_mask_t.permute(0,2,1)  # shape: [B, C, H*W]
            cam2_mask = cam2_mask_t.permute(0,2,1)
            # exclude pos/neg proto class with few cam samples
            label1[cam1_mask.sum(-1) < 0.01*H1*W1] = torch.tensor(0.).cuda()
            label2[cam2_mask.sum(-1) < 0.01*H2*W2] = torch.tensor(0.).cuda()

            cam1_mask_B_sum = (cam1_mask.sum(1)>0).sum(-1)  # shape: [B]
            cam2_mask_B_sum = (cam2_mask.sum(1)>0).sum(-1)
            self.mask_area = cam1_mask_B_sum.float().mean()
        # class-wise average features
        feat1_mask_mean = torch.bmm(cam1_mask, feat1_flatten_t) / (torch.sum(cam1_mask, 2, keepdim=True)+1e-8) # shape: [B, C, 512]
        feat2_mask_mean = torch.bmm(cam2_mask, feat2_flatten_t) / (torch.sum(cam2_mask, 2, keepdim=True)+1e-8)
        feat1_mask_mean_norm_t = F.normalize(feat1_mask_mean,dim=2).transpose(1,2)   # shape: [B, 512, C]
        feat2_mask_mean_norm_t = F.normalize(feat2_mask_mean,dim=2).transpose(1,2)

        with torch.no_grad():
            intraclass_mask_1 = cam1_mask_t * label2.unsqueeze(1)    # shape: [B, H*W, C]
            intraclass_mask_2 = cam2_mask_t * label1.unsqueeze(1)    # shape: [B, H*W, C]
            # exclude invalid pixels
            cam1_valid_cls[intraclass_mask_1.sum(2) < 1] = 255
            cam2_valid_cls[intraclass_mask_2.sum(2) < 1] = 255
        # pixel-wise features
        feat1_flatten_t_norm = F.normalize(feat1_flatten_t,dim=2)
        feat2_flatten_t_norm = F.normalize(feat2_flatten_t,dim=2)
        # infoNCE loss
        sim_12 = torch.matmul(feat1_flatten_t_norm, feat2_mask_mean_norm_t) / self.temperature # shape: [B, H*W, C]
        sim_21 = torch.matmul(feat2_flatten_t_norm, feat1_mask_mean_norm_t) / self.temperature # shape: [B, H*W, C]
        criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
        loss_12 = criterion(sim_12.view(-1, C), cam1_valid_cls.view(-1))
        loss_21 = criterion(sim_21.view(-1, C), cam2_valid_cls.view(-1))
        loss = loss_12 / 2 + loss_21 / 2
        return loss
    def area(self):
        return self.mask_area

class L_holift(nn.Module):
    def __init__(self):
        super(L_holift, self).__init__()
        
    def forward(self, cam, label):
        cam = F.relu(cam)
        # exclude unlabelled class
        cam = cam * label.unsqueeze(-1).unsqueeze(-1)
        # normalize cam
        cam = cam / (cam.amax((-1,-2)).unsqueeze(-1).unsqueeze(-1)+1e-8)

        cam_flatten = cam.reshape(cam.shape[0], cam.shape[1], -1)   # shape: [B, C, H*W]
        # total activation values across classes
        cam_sum = cam_flatten.sum(1)
        loss = torch.mean(torch.square(cam_sum - 1))
        return loss 
