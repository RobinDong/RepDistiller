# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


class AFD(nn.Module):
    def __init__(self, args):
        super(AFD, self).__init__()
        self.guide_layers = args.guide_layers
        self.hint_layers = args.hint_layers
        self.attention = Attention(args)

    def forward(self, g_s, g_t):
        g_t = [g_t[i] for i in self.guide_layers]
        g_s = [g_s[i] for i in self.hint_layers]
        loss = self.attention(g_s, g_t)
        return sum(loss)


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim
        self.n_t = args.n_t
        
        # Advanced AFD improvements
        self.temperature = float(getattr(args, 'afd_temp', 2.0))  # Higher temperature for softer attention
        self.entropy_lambda = float(getattr(args, 'afd_entropy_lambda', 0.1))  # Entropy regularization
        self.use_cosine = bool(getattr(args, 'afd_use_cosine', True))  # Use cosine similarity
        self.layer_weighting = getattr(args, 'afd_layer_weighting', 'learned')  # 'uniform' | 'sqrt' | 'learned'
        self.attn_dropout = nn.Dropout(float(getattr(args, 'afd_attn_dropout', 0.1)))  # Attention dropout
        self.focal_alpha = float(getattr(args, 'afd_focal_alpha', 2.0))  # Focal loss alpha
        self.use_adaptive_pooling = bool(getattr(args, 'afd_adaptive_pooling', True))
        
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)
        
        # Learnable layer weights for adaptive weighting
        if self.layer_weighting == 'learned':
            self.layer_weights = nn.Parameter(torch.ones(len(args.t_shapes)))
            nn.init.uniform_(self.layer_weights, 0.8, 1.2)
        
        # Feature normalization layers
        self.feature_norm = nn.ModuleList([
            nn.LayerNorm(s_shape[1]) for s_shape in args.s_shapes
        ])

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        # Enhanced attention computation with cosine similarity
        if self.use_cosine:
            # Normalize for cosine similarity
            bilinear_key_norm = F.normalize(bilinear_key, dim=-1)
            query_norm = F.normalize(query, dim=-1)
            cosine_sim = torch.einsum('bstq,btq->bts', bilinear_key_norm, query_norm)
            logit = torch.add(cosine_sim, p_logit)
        else:
            logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            logit = logit / self.temperature
            
        atts = F.softmax(logit, dim=2)  # b x t x s
        atts = self.attn_dropout(atts)
        
        # Calculate entropy regularization
        entropy_loss = 0.0
        if self.entropy_lambda > 0.0:
            entropy = -torch.sum(atts * torch.log(atts + 1e-8), dim=2).mean()
            entropy_loss = self.entropy_lambda * entropy
        
        # Prepare layer-wise weights
        t_layers = atts.size(1)
        if self.layer_weighting == 'uniform':
            weights = atts.new_ones(t_layers) / float(t_layers)
        elif self.layer_weighting == 'sqrt':
            weights = atts.new_ones(t_layers) / np.sqrt(t_layers)
        elif self.layer_weighting == 'learned':
            weights = F.softmax(self.layer_weights, dim=0)
        else:
            weights = atts.new_ones(t_layers)
            
        loss = []
        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            # Apply feature normalization
            if i < len(self.feature_norm):
                h_hat_s = self.feature_norm[i](h_hat_s.transpose(1, 2)).transpose(1, 2)
            
            diff = self.cal_diff_enhanced(h_hat_s, h_t, atts[:, i], weights[i])
            loss.append(diff)
            
        total_loss = sum(loss) + entropy_loss
        return [total_loss]  # Return as list for compatibility

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff
    
    def cal_diff_enhanced(self, v_s, v_t, att, layer_weight):
        """Enhanced difference calculation with multiple loss functions"""
        # Smooth L1 loss (Huber loss) - more robust to outliers
        diff_smooth = F.smooth_l1_loss(v_s, v_t.unsqueeze(1), reduction='none').mean(2)
        
        # Focal loss for hard examples
        diff_mse = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        focal_weight = torch.pow(diff_mse, self.focal_alpha - 1)
        diff_focal = focal_weight * diff_mse
        
        # Combine losses
        diff = 0.7 * diff_smooth + 0.3 * diff_focal
        
        # Apply attention weighting and layer weighting
        diff = torch.mul(diff, att).sum(1).mean() * layer_weight
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)
        self.use_adaptive_pooling = bool(getattr(args, 'afd_adaptive_pooling', True))
        
        # Enhanced samplers with multiple pooling strategies
        if self.use_adaptive_pooling:
            self.samplers = nn.ModuleList([SampleEnhanced(t_shape) for t_shape in args.unique_t_shapes])
        else:
            self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])

        # Enhanced key layers with residual connections
        self.key_layer = nn.ModuleList([
            nn.Sequential(
                nn_bn_relu(s_shape[1], self.qk_dim),
                nn.Dropout(0.1)
            ) for s_shape in args.s_shapes
        ])
        
        # Multi-scale bilinear transformation
        self.bilinear = nn.Sequential(
            nn_bn_relu(args.qk_dim, args.qk_dim * 2),
            nn.ReLU(inplace=False),
            nn_bn_relu(args.qk_dim * 2, args.qk_dim * len(args.t_shapes))
        )

    def forward(self, g_s):
        bs = g_s[0].size(0)
        
        # Enhanced channel mean with multiple pooling strategies
        channel_mean = []
        for f_s in g_s:
            # Global average pooling
            gap = f_s.mean(3).mean(2)
            # Global max pooling
            gmp = f_s.max(3)[0].max(2)[0]
            # Combine GAP and GMP
            channel_mean.append(0.7 * gap + 0.3 * gmp)
            
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        # Enhanced key generation with residual connections
        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1).view(bs * self.s, -1)  # Bs x h
        
        # Multi-scale bilinear transformation
        bilinear_key = self.bilinear(key).view(bs, self.s, self.t, -1)
        
        # Enhanced value normalization
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s


class SampleEnhanced(nn.Module):
    """Enhanced sampling with multiple pooling strategies and attention"""
    def __init__(self, t_shape):
        super(SampleEnhanced, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.avg_pool = nn.AdaptiveAvgPool2d((t_H, t_W))
        self.max_pool = nn.AdaptiveMaxPool2d((t_H, t_W))
        
        # Channel attention for adaptive weighting
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, g_s, bs):
        results = []
        for f_s in g_s:
            # Power normalization
            f_power = f_s.pow(2).mean(1, keepdim=True)
            
            # Channel attention
            attn = self.channel_attn(f_power)
            f_attended = f_power * attn
            
            # Multi-scale pooling
            avg_feat = self.avg_pool(f_attended)
            max_feat = self.max_pool(f_attended)
            
            # Combine features
            combined = 0.7 * avg_feat + 0.3 * max_feat
            results.append(combined.view(bs, -1))
            
        return torch.stack(results, dim=1)
