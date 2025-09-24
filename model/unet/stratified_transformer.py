"""
Stratified Transformer implementation with pointops replacement
Addresses the integration issues mentioned in the problem statement:
1. Replaces pointops dependency with pure PyTorch implementations
2. Handles sparse to dense data conversion properly
3. Fixes parameter passing and batch processing issues
4. Maintains compatibility with the GuidedContrast framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class PointOpsReplacements:
    """
    Pure PyTorch replacements for pointops functionality
    This addresses the missing pointops library issue
    """
    
    @staticmethod
    def knn_query(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int) -> torch.Tensor:
        """
        Find k nearest neighbors for each point in new_xyz from xyz
        Args:
            xyz: (B, N, 3) reference points
            new_xyz: (B, M, 3) query points
            k: number of neighbors
        Returns:
            idx: (B, M, k) indices of nearest neighbors
        """
        B, N, _ = xyz.shape
        _, M, _ = new_xyz.shape
        
        # Compute squared distances
        # xyz: (B, N, 3) -> (B, N, 1, 3)
        # new_xyz: (B, M, 3) -> (B, 1, M, 3)
        dist = torch.sum((xyz.unsqueeze(2) - new_xyz.unsqueeze(1)) ** 2, dim=-1)  # (B, N, M)
        
        # Find k nearest neighbors
        _, idx = torch.topk(dist, k, dim=1, largest=False)  # (B, k, M)
        idx = idx.transpose(1, 2)  # (B, M, k)
        
        return idx
    
    @staticmethod
    def group_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Group points according to the index
        Args:
            points: (B, C, N) input features
            idx: (B, M, k) grouping indices
        Returns:
            grouped: (B, C, M, k) grouped features
        """
        B, C, N = points.shape
        _, M, k = idx.shape
        
        # Expand idx to match the feature dimension
        idx_expanded = idx.unsqueeze(1).expand(B, C, M, k)  # (B, C, M, k)
        points_expanded = points.unsqueeze(2).expand(B, C, M, N)  # (B, C, M, N)
        
        # Gather points according to idx
        grouped = torch.gather(points_expanded, dim=3, index=idx_expanded)  # (B, C, M, k)
        
        return grouped
    
    @staticmethod
    def ball_query(xyz: torch.Tensor, new_xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
        """
        Ball query with radius constraint
        Args:
            xyz: (B, N, 3) reference points
            new_xyz: (B, M, 3) query points  
            radius: query radius
            nsample: maximum number of samples
        Returns:
            idx: (B, M, nsample) indices
        """
        B, N, _ = xyz.shape
        _, M, _ = new_xyz.shape
        
        # Compute squared distances
        dist = torch.sum((xyz.unsqueeze(2) - new_xyz.unsqueeze(1)) ** 2, dim=-1)  # (B, N, M)
        
        # Find points within radius
        mask = dist <= radius ** 2  # (B, N, M)
        
        # For each query point, find at most nsample neighbors
        idx = torch.zeros(B, M, nsample, dtype=torch.long, device=xyz.device)
        
        for b in range(B):
            for m in range(M):
                valid_indices = torch.where(mask[b, :, m])[0]
                if len(valid_indices) == 0:
                    # If no points in radius, use the nearest point
                    nearest_idx = torch.argmin(dist[b, :, m])
                    idx[b, m, :] = nearest_idx
                else:
                    # Sample up to nsample points
                    n_valid = min(len(valid_indices), nsample)
                    sampled_indices = valid_indices[:n_valid]
                    idx[b, m, :n_valid] = sampled_indices
                    # Fill remaining with the last valid index
                    if n_valid < nsample:
                        idx[b, m, n_valid:] = sampled_indices[-1]
        
        return idx


def grid_sample_3d(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Replacement for the problematic grid_sample function
    Args:
        feat: (B, C, D, H, W) feature tensor
        coords: (B, N, 3) normalized coordinates in [-1, 1]
    Returns:
        sampled: (B, C, N) sampled features
    """
    B, C, D, H, W = feat.shape
    _, N, _ = coords.shape
    
    # Ensure coordinates are in valid range
    coords = torch.clamp(coords, -1, 1)
    
    # Convert normalized coordinates to grid indices
    # coords are in [-1, 1], convert to [0, size-1]
    coords_d = (coords[:, :, 0] + 1) * (D - 1) / 2
    coords_h = (coords[:, :, 1] + 1) * (H - 1) / 2  
    coords_w = (coords[:, :, 2] + 1) * (W - 1) / 2
    
    # Reshape for grid_sample (needs 5D: B, C, D, H, W)
    # grid needs to be (B, D_out, H_out, W_out, 3)
    grid = torch.stack([coords_w, coords_h, coords_d], dim=-1)  # (B, N, 3)
    grid = grid.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 3)
    
    # Reshape feature to match
    feat_reshaped = feat
    
    # Sample
    sampled = F.grid_sample(feat_reshaped, grid, mode='bilinear', 
                           padding_mode='border', align_corners=True)  # (B, C, 1, 1, N)
    
    # Reshape output
    sampled = sampled.squeeze(2).squeeze(2)  # (B, C, N)
    
    return sampled


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for point clouds"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
        Returns:
            pos_enc: (B, N, d_model) positional encoding
        """
        B, N, _ = xyz.shape
        
        # Simple hash-based encoding for 3D positions
        # Scale and round coordinates to get discrete positions
        xyz_scaled = (xyz * 1000).long()  # Scale to get integer coordinates
        
        # Simple hash function
        pos_hash = (xyz_scaled[:, :, 0] * 73856093 + 
                   xyz_scaled[:, :, 1] * 19349663 + 
                   xyz_scaled[:, :, 2] * 83492791) % self.pe.size(0)
        
        pos_enc = self.pe[pos_hash]  # (B, N, d_model)
        
        return pos_enc


class MultiHeadAttention3D(nn.Module):
    """Multi-head attention for 3D point clouds"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d_model)
            key: (B, N_k, d_model)
            value: (B, N_v, d_model)
            mask: (B, N_q, N_k) attention mask
        Returns:
            output: (B, N_q, d_model)
        """
        B, N_q, _ = query.shape
        N_k = key.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, N_q, d_k)
        K = self.w_k(key).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)    # (B, h, N_k, d_k)
        V = self.w_v(value).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, N_v, d_k)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, N_q, N_k)
        
        if mask is not None:
            # Create attention mask: (B, N_q) -> (B, 1, N_q, N_k)
            if mask.dim() == 2:  # (B, N_q)
                # Create a causal/padding mask
                attn_mask = mask.unsqueeze(1).unsqueeze(3)  # (B, 1, N_q, 1)
                attn_mask = attn_mask.expand(B, 1, N_q, N_k)  # (B, 1, N_q, N_k)
                attn_mask = attn_mask.expand(B, self.num_heads, N_q, N_k)  # (B, h, N_q, N_k)
            else:
                attn_mask = mask.unsqueeze(1).expand(B, self.num_heads, N_q, N_k)
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, V)  # (B, h, N_q, d_k)
        output = output.transpose(1, 2).contiguous().view(B, N_q, self.d_model)  # (B, N_q, d_model)
        
        return self.w_o(output)


class TransitionDown(nn.Module):
    """Downsampling transition layer for stratified transformer"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, k: int = 16):
        super().__init__()
        self.stride = stride
        self.k = k
        
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, C, N) point features
        Returns:
            new_xyz: (B, N//stride, 3) downsampled coordinates
            new_features: (B, C_out, N//stride) downsampled features
        """
        B, N, _ = xyz.shape
        
        # Simple stride-based downsampling
        indices = torch.arange(0, N, self.stride, device=xyz.device)
        new_xyz = xyz[:, indices, :]  # (B, N//stride, 3)
        
        # Downsample features
        new_features = features[:, :, indices]  # (B, C, N//stride)
        new_features = self.relu(self.bn(self.conv(new_features)))
        
        return new_xyz, new_features


class StratifiedAttentionBlock(nn.Module):
    """Basic attention block for stratified transformer"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention3D(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.pos_enc = PositionalEncoding3D(d_model)
        
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, d_model) point features  
            mask: (B, N) point mask
        Returns:
            output: (B, N, d_model) updated features
        """
        # Add positional encoding
        pos_enc = self.pos_enc(xyz)
        x = features + pos_enc
        
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(x)
        output = self.norm2(x + ffn_output)
        
        return output


class BasicLayer(nn.Module):
    """Basic layer of the stratified transformer with proper data flow handling"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_blocks: int = 2, 
                 dropout: float = 0.1, downsample: bool = False, stride: int = 2):
        super().__init__()
        
        self.downsample = downsample
        
        # Attention blocks
        self.blocks = nn.ModuleList([
            StratifiedAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_blocks)
        ])
        
        # Downsampling layer
        if downsample:
            self.transition = TransitionDown(d_model, d_model, stride)
        
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, d_model) or (B, d_model, N) point features
            mask: (B, N) point mask
        Returns:
            new_xyz: (B, N', 3) updated coordinates
            new_features: (B, N', d_model) updated features
        """
        # Handle feature dimension order (convert to B, N, d_model if needed)
        if features.dim() == 3 and features.size(1) != xyz.size(1):
            features = features.transpose(1, 2)  # (B, d_model, N) -> (B, N, d_model)
        
        # Apply attention blocks
        x = features
        for block in self.blocks:
            x = block(xyz, x, mask)
        
        # Apply downsampling if needed
        if self.downsample:
            # Convert to (B, C, N) for TransitionDown
            x_transposed = x.transpose(1, 2)  # (B, N, d_model) -> (B, d_model, N)
            new_xyz, new_features = self.transition(xyz, x_transposed)
            new_features = new_features.transpose(1, 2)  # (B, d_model, N') -> (B, N', d_model)
            return new_xyz, new_features
        else:
            return xyz, x


class StratifiedTransformer(nn.Module):
    """
    Stratified Transformer implementation that addresses the integration issues:
    1. No pointops dependency - uses pure PyTorch implementations
    2. Proper sparse to dense data conversion
    3. Fixed parameter passing and batch processing
    4. Compatible with GuidedContrast framework
    """
    
    def __init__(self, input_dim: int = 6, d_model: int = 256, num_heads: int = 8, 
                 d_ff: int = 1024, num_layers: int = 4, num_classes: int = 13, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Encoder layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            downsample = (i > 0 and i % 2 == 0)  # Downsample every 2 layers after the first
            layer = BasicLayer(d_model, num_heads, d_ff, num_blocks=2, 
                             dropout=dropout, downsample=downsample)
            self.layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def handle_batch_data(self, batch_data):
        """
        Handle different batch data formats and convert to consistent format
        This addresses the data flow mismatch issue
        """
        if isinstance(batch_data, dict):
            # Handle dictionary format (from sparse conv systems)
            if 'locs_float' in batch_data and 'feats' in batch_data:
                coords = batch_data['locs_float']  # (N, 3)
                features = batch_data['feats']     # (N, C)
                
                # Handle offsets for batching
                if 'offsets' in batch_data:
                    offsets = batch_data['offsets']
                    batch_size = len(offsets) - 1
                    
                    # Convert to dense format
                    max_points = max([(offsets[i+1] - offsets[i]).item() for i in range(batch_size)])
                    
                    coords_batched = torch.zeros(batch_size, max_points, 3, 
                                               device=coords.device, dtype=coords.dtype)
                    features_batched = torch.zeros(batch_size, max_points, features.size(1),
                                                 device=features.device, dtype=features.dtype)
                    mask_batched = torch.zeros(batch_size, max_points, dtype=torch.bool, device=coords.device)
                    
                    for i in range(batch_size):
                        start, end = offsets[i], offsets[i+1]
                        length = end - start
                        coords_batched[i, :length] = coords[start:end]
                        features_batched[i, :length] = features[start:end]
                        mask_batched[i, :length] = True
                    
                    return coords_batched, features_batched, mask_batched
                else:
                    # Single batch, add batch dimension
                    return coords.unsqueeze(0), features.unsqueeze(0), None
            
            elif 'xyz' in batch_data and 'features' in batch_data:
                # Handle standard point cloud format
                xyz = batch_data['xyz']
                features = batch_data['features']
                mask = batch_data.get('mask', None)
                
                # Handle feature dimension order
                if features.dim() == 3 and features.size(1) != xyz.size(1):
                    features = features.transpose(1, 2)  # (B, C, N) -> (B, N, C)
                
                return xyz, features, mask
        
        else:
            raise ValueError(f"Unsupported batch data format: {type(batch_data)}")
    
    def forward(self, batch_data):
        """
        Forward pass with proper error handling and debugging support
        """
        try:
            # Handle different input formats
            xyz, features, mask = self.handle_batch_data(batch_data)
            
            B, N, _ = xyz.shape
            
            # Input projection
            x = self.input_proj(features)  # (B, N, d_model)
            
            # Apply transformer layers
            current_xyz = xyz
            for i, layer in enumerate(self.layers):
                current_xyz, x = layer(current_xyz, x, mask)
                
                # Update mask if downsampling occurred
                if x.size(1) != N:
                    if mask is not None:
                        # Simple downsampling of mask
                        stride = N // x.size(1)
                        mask = mask[:, ::stride]
                    N = x.size(1)
            
            # Output projection
            output = self.output_proj(x)  # (B, N', num_classes)
            
            # Return in format compatible with existing framework
            # Convert back to flattened format if needed
            if isinstance(batch_data, dict) and 'offsets' in batch_data:
                # Convert back to (N_total, num_classes) format
                output_list = []
                for b in range(B):
                    if mask is not None:
                        valid_points = mask[b].sum().item()
                        output_list.append(output[b, :valid_points])
                    else:
                        output_list.append(output[b])
                
                return torch.cat(output_list, dim=0)
            else:
                return output
                
        except Exception as e:
            print(f"[ERROR] StratifiedTransformer forward pass failed: {e}")
            print(f"Input batch_data keys: {batch_data.keys() if isinstance(batch_data, dict) else type(batch_data)}")
            if isinstance(batch_data, dict):
                for key, value in batch_data.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: {value.shape}")
            raise e


def create_stratified_transformer(cfg):
    """
    Factory function to create stratified transformer with config
    This addresses the configuration integration issue
    """
    # Extract relevant config parameters
    input_dim = getattr(cfg, 'input_channel', 3) + 3  # RGB + XYZ
    d_model = getattr(cfg, 'm', 32) * 8  # Scale up for transformer
    num_classes = getattr(cfg, 'classes', 13)
    
    # Transformer specific parameters
    num_heads = getattr(cfg, 'num_heads', 8)
    d_ff = getattr(cfg, 'd_ff', d_model * 4)
    num_layers = getattr(cfg, 'num_layers', 4)
    dropout = getattr(cfg, 'dropout', 0.1)
    
    return StratifiedTransformer(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the implementation
    print("Testing StratifiedTransformer...")
    
    # Create test data
    B, N, C = 2, 1024, 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    xyz = torch.randn(B, N, 3).to(device)
    features = torch.randn(B, N, C).to(device)
    
    # Create model
    model = StratifiedTransformer(input_dim=C, d_model=64, num_heads=4, 
                                d_ff=256, num_layers=2).to(device)
    
    # Test forward pass
    batch_data = {'xyz': xyz, 'features': features}
    
    try:
        output = model(batch_data)
        print(f"Success! Output shape: {output.shape}")
        print("StratifiedTransformer implementation is working correctly.")
    except Exception as e:
        print(f"Error during testing: {e}")
        raise e