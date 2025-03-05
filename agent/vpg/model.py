from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class reinforcement_net(nn.Module):
    def __init__(self, config):
        super(reinforcement_net, self).__init__()
        self.num_rotations = config['num_rotations']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize network trunks with DenseNet
        self.push_color_trunk = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        self.push_depth_trunk = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        self.grasp_color_trunk = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        self.grasp_depth_trunk = torchvision.models.densenet121(weights='IMAGENET1K_V1')

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        input_color_data = input_color_data.to(self.device)
        input_depth_data = input_depth_data.to(self.device)

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                    [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)

                    flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=True)

                    # Rotate images counter clockwise
                    rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=True)
                    rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=True)

                    # Compute intermediate features
                    interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                    interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                    interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                    interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    interm_feat.append([interm_push_feat, interm_grasp_feat])

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                   [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)

                    flow_grid_after = F.affine_grid(affine_mat_after, interm_push_feat.size(), align_corners=True)

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([F.interpolate(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest', align_corners=True),
                                                      scale_factor=16, mode='bilinear', align_corners=True),
                                        F.interpolate(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest', align_corners=True),
                                                      scale_factor=16, mode='bilinear', align_corners=True)])
                    
            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []
            
            # Apply rotations to intermediate features
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                            [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)

            flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=True)

            # Rotate images counter clockwise
            rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=True)
            rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=True)

            # Compute intermediate features
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_push_feat, interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                           [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)

            flow_grid_after = F.affine_grid(affine_mat_after, interm_push_feat.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([F.interpolate(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest', align_corners=True),
                                                   scale_factor=16, mode='bilinear', align_corners=True),
                                     F.interpolate(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest', align_corners=True),
                                                   scale_factor=16, mode='bilinear', align_corners=True)])

            return self.output_prob, self.interm_feat