import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_channels, reduction_ratio=16):  # Increased reduction ratio
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Improved MLP with GELU and proper initialization
        reduced_dim = max(in_channels // reduction_ratio, 4)  # Smaller minimum
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_dim),
            nn.GELU(),  # Better than ReLU for this task
            nn.Dropout(0.1),  # Add regularization
            nn.Linear(reduced_dim, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

        # Better initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1) * x

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=5):  # Reduced kernel size
        super(SpatialAttention, self).__init__()
        # Use GroupNorm instead of no normalization
        self.conv = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size, padding=kernel_size//2, bias=False),  # Reduced channels
            nn.GroupNorm(2, 4),
            nn.GELU(),
            nn.Conv2d(4, 1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # Better initialization
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(x_cat)
        return self.sigmoid(attention) * x

class FlexiFilter(nn.Module):
    """FlexiFilter with better learning dynamics"""
    def __init__(self, kernel_size=5, in_channels=1, out_channels=1):
        super(FlexiFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main convolution with better initialization
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding)
        
        # Learnable parameters with better initialization
        self.alpha = nn.Parameter(torch.tensor(0.8))  # Higher initial value
        self.beta = nn.Parameter(torch.tensor(0.2))
        self.adaptive_scale = nn.Parameter(torch.tensor(0.2))  # Higher initial scale
        
        # Processing layers with GroupNorm
        self.process = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(4, out_channels), out_channels),  # Smaller groups
            nn.GELU(),  # Better activation
            nn.Dropout2d(0.05)  # Reduced dropout
        )

        # Better initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        # Standard convolution
        conv_out = self.conv(x)
        
        # Apply learnable transformations
        processed = self.process(conv_out)
        
        # Residual connection with learnable mixing
        if x.shape[1] == processed.shape[1]:
            return self.alpha * x + self.beta * processed * self.adaptive_scale
        else:
            # Better channel matching
            if self.out_channels < self.in_channels:
                # Channel reduction - use 1x1 conv
                channel_matcher = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False).to(x.device)
                nn.init.xavier_normal_(channel_matcher.weight)
                x_matched = channel_matcher(x)
            else:
                # Channel expansion - pad with zeros
                pad_channels = self.out_channels - self.in_channels
                x_matched = F.pad(x, (0, 0, 0, 0, 0, pad_channels))
            
            return self.alpha * x_matched + self.beta * processed * self.adaptive_scale

class BoundaryExtractor(nn.Module):
    """Boundary extractor with better edge detection"""
    def __init__(self, in_channels=1):
        super(BoundaryExtractor, self).__init__()
        
        # Edge detection with more sophisticated kernels
        self.edge_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)  # Reduced from 12
        
        # Initialize with improved edge detection patterns
        with torch.no_grad():
            kernels = torch.zeros(8, in_channels, 3, 3)
            
            # Sobel operators (better than simple gradients)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            # Standard edge patterns with Sobel
            kernels[0, 0] = sobel_x  # Horizontal Sobel
            kernels[1, 0] = sobel_y  # Vertical Sobel
            
            # Diagonal Sobel
            kernels[2, 0] = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32)
            kernels[3, 0] = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
            
            # Laplacian variants
            kernels[4, 0] = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
            kernels[5, 0] = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            
            # Additional patterns
            kernels[6, 0] = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            kernels[7, 0] = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
            
            self.edge_conv.weight.data = kernels
        
        # Combination layers - smaller network
        self.combine = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Reduced from 24
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),  # Reduced from 8
            nn.GroupNorm(4, 4),
            nn.GELU(),
            nn.Conv2d(4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Better initialization
        for m in self.combine:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        edges = F.gelu(self.edge_conv(x))  # GELU instead of LeakyReLU
        return self.combine(edges)

class FlexiBlock(nn.Module):
    """FlexiBlock with better regularization"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(FlexiBlock, self).__init__()
        self.flexi_filter = FlexiFilter(kernel_size, in_channels, out_channels)
        self.batch_norm = nn.GroupNorm(min(4, out_channels), out_channels)  # Smaller groups
        self.activation = nn.GELU()  # Better activation
        self.dropout = nn.Dropout2d(0.05)  # Reduced dropout
        
    def forward(self, x):
        x = self.flexi_filter(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.dropout(x)

class FlexiCell(nn.Module):
    """FlexiCell with improved architecture"""
    def __init__(self, n_channels=1, n_classes=2):
        super(FlexiCell, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Reduced base channels for memory efficiency
        base_channels = 32  # Reduced from 48
        kernel_sizes = [7, 5, 3]  # Reduced number of scales from [9, 7, 5, 3]
        
        # Initial projection
        self.init_conv = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Dropout2d(0.05),
        )
        
        # Processing blocks
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i, kernel_size in enumerate(kernel_sizes):
            out_channels = base_channels * (2 ** min(i, 2))  # Cap at 4x instead of 8x
            
            block = nn.Sequential(
                FlexiFilter(kernel_size, current_channels, out_channels),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
                nn.Dropout2d(0.05),  # Reduced dropout
                ChannelAttention(out_channels, reduction_ratio=16),  # Increased reduction
                SpatialAttention(kernel_size=5),  # Reduced kernel size
            )
            self.blocks.append(block)
            current_channels = out_channels
        
        # Boundary processing
        self.boundary = BoundaryExtractor(n_channels)
        
        # Reduced feature fusion
        total_channels = sum([base_channels * (2 ** min(i, 2)) for i in range(len(kernel_sizes))]) + 1
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 128, 3, padding=1),  # Reduced from 256
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 64, 3, padding=1),  # Reduced from 128
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(64, 32, 3, padding=1),  # Reduced from 64
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        
        # Output heads with smaller architecture
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # Reduced from 32
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(16, n_classes, 1)
        )
        
        self.edge_head = nn.Sequential(
            nn.Conv2d(32, 8, 3, padding=1),  # Reduced from 16
            nn.GroupNorm(4, 8),
            nn.GELU(),
            nn.Conv2d(8, n_classes, 1)
        )

        # Apply better weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Better weight initialization"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Initial projection
        features = self.init_conv(x)
        
        # Store input for boundary processing
        input_for_boundary = x
        
        # Collect features from each block
        all_features = []
        current_x = features
        
        for block in self.blocks:
            current_x = block(current_x)
            all_features.append(current_x)
        
        # Boundary processing
        boundary = self.boundary(input_for_boundary)
        
        # Ensure all features have the same spatial dimensions
        target_size = all_features[0].shape[2:]
        aligned_features = []
        
        for feat in all_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # Align boundary map
        if boundary.shape[2:] != target_size:
            boundary = F.interpolate(boundary, size=target_size, mode='bilinear', align_corners=False)
        
        # Combine all features
        combined = torch.cat(aligned_features + [boundary], dim=1)
        
        # Final processing
        fused_features = self.fusion(combined)
        
        # Generate outputs
        segmentation = self.seg_head(fused_features)
        edges = self.edge_head(fused_features)
        
        return segmentation, edges

class LiteFlexiCell(nn.Module):
    """Lightweight version"""
    def __init__(self, n_channels=1, n_classes=2):
        super(LiteFlexiCell, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Better lightweight architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1),  # Reduced from 24
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),  # Reduced from 48
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1),  # Reduced from 96
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        
        # Boundary extractor
        self.boundary_extractor = BoundaryExtractor(n_channels)
        
        # Better final processing
        self.final_layers = nn.Sequential(
            nn.Conv2d(48 + 1, 32, kernel_size=3, padding=1),  # Reduced from 64
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Reduced from 32
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        
        # Output layers
        self.output_seg = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),  # Reduced from 16
            nn.GELU(),
            nn.Conv2d(8, n_classes, 1)
        )
        
        self.output_edges = nn.Sequential(
            nn.Conv2d(16, 4, 3, padding=1),  # Reduced from 8
            nn.GELU(),
            nn.Conv2d(4, n_classes, 1)
        )

        # Apply better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Better weight initialization"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        # Main feature extraction
        features = self.encoder(x)
        
        # Extract boundaries
        boundary_map = self.boundary_extractor(x)
        
        # Ensure dimensions match
        if boundary_map.shape[2:] != features.shape[2:]:
            boundary_map = F.interpolate(boundary_map, size=features.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine features
        combined = torch.cat([features, boundary_map], dim=1)
        
        # Final processing
        features = self.final_layers(combined)
        
        # Generate outputs
        segmentation = self.output_seg(features)
        edges = self.output_edges(features)
        
        return segmentation, edges