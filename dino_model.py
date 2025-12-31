import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from tools.deit_features import deit_tiny_patch_features, deit_small_patch_features, dinov3_patch_features
from tools.cait_features import cait_xxs24_224_features

base_architecture_to_features = {'deit_small_patch16_224': deit_small_patch_features,
                                 'deit_tiny_patch16_224': deit_tiny_patch_features,
                                 #'deit_base_patch16_224':deit_base_patch16_224,
                                 'cait_xxs24_224': cait_xxs24_224_features,
                                 'dinov3': dinov3_patch_features}

class PPNetDINO(nn.Module):
    """Fixed version addressing DINOv3-specific issues"""
    
    def __init__(self, features, img_size, prototype_shape,
                 num_classes, init_weights=True,
                 prototype_activation_function='log',
                 sig_temp=1.0,
                 radius=3,
                 add_on_layers_type='bottleneck',
                 layers=None,
                 # NEW PARAMETERS FOR DINOV3
                 dinov3_feature_scale=10.0,  # CRITICAL: scale factor for gradients
                 init_prototypes_from_data=True,
                 prototype_init_std=0.02):  # Better initialization

        super(PPNetDINO, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape  # (num_prototypes, dim, n_subpatches)
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.epsilon = 1e-4
        self.normalizer = nn.Softmax(dim=1)
        self.layers = layers
        self.warmup = False
        self.patch_size = 16  # DINOv3-ViT-L/16
        self.spatial_size = img_size // self.patch_size
        self.num_spatial_patches = self.spatial_size ** 2
        self.prototype_activation_function = prototype_activation_function
        
        # DINOv3-SPECIFIC PARAMETERS
        self.dinov3_feature_scale = dinov3_feature_scale
        self.init_prototypes_from_data = init_prototypes_from_data
        
        assert(self.num_prototypes % self.num_classes == 0)
        
        # Prototype class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        
        self.features = features
        
        # FIX 1: Better prototype initialization
        # ALWAYS initialize with random values first
        self.prototype_vectors = nn.Parameter(
            torch.randn(self.prototype_shape) * prototype_init_std,
            requires_grad=True
        )
        # Normalize after initialization to unit vectors
        with torch.no_grad():
            for i in range(self.prototype_shape[-1]):
                self.prototype_vectors.data[:, :, i] = F.normalize(
                    self.prototype_vectors.data[:, :, i], p=2, dim=1
                )
        
        self.prototypes_initialized = not init_prototypes_from_data
        self.init_prototypes_from_data = init_prototypes_from_data
        
        self.radius = radius
        self.patch_select = nn.Parameter(
            torch.ones(1, prototype_shape[0], prototype_shape[-1]) * 0.1,
            requires_grad=True
        )
        self.temp = sig_temp
        self.ones = nn.Parameter(
            torch.ones(self.prototype_shape),
            requires_grad=False
        )
        
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        
        # Detect architecture
        features_name = str(features).upper()
        if features_name.startswith('VISION'):
            self.arc = 'deit'
        elif features_name.startswith('CAIT'):
            self.arc = 'cait'
        elif features_name.startswith('DINO'):
            self.arc = 'dinov3'
        else:
            self.arc = 'unknown'
        
        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        """Extract features from DINOv3 backbone"""
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)
        
        if self.arc == 'deit':
            x = torch.cat((cls_token, x), dim=1)
            x = self.features.pos_drop(x + self.features.pos_embed)
            x = self.features.blocks(x)
            x = self.features.norm(x)
        elif self.arc == 'dinov3':
            x = x.reshape(x.size(0), -1, x.size(-1))
            x = torch.cat((cls_token, x), dim=1)
            for blk in self.features.blocks:
                x = blk(x)
            x = self.features.norm(x)
        
        # FIX 2: Remove CLS token subtraction - this creates instability
        # OLD: x_2 = x[:, 1:] - x[:, 0].unsqueeze(1)
        # NEW: Just use patch tokens directly
        x_2 = x[:, 1:]  # [bsz, num_patches, dim]
        
        fea_len = x_2.shape[1]
        B = x_2.shape[0]
        fea_width = int(fea_len ** 0.5)
        fea_height = int(fea_len ** 0.5)
        feature_emb = x_2.permute(0, 2, 1).reshape(B, -1, fea_width, fea_height)
        
        return feature_emb

    def _cosine_convolution(self, x):
        """Compute cosine similarity between features and prototypes"""
        x = F.normalize(x, p=2, dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        distances = -distances
        return distances

    def _project2basis(self, x):
        """Project features onto prototype basis"""
        x = F.normalize(x, p=2, dim=1)
        
        # Normalize prototypes per subpatch
        now_prototype_vectors = torch.zeros_like(self.prototype_vectors)
        for i in range(self.prototype_shape[-1]):
            now_prototype_vectors[:, :, i] = F.normalize(
                self.prototype_vectors[:, :, i], p=2, dim=1
            )
        
        # Compute cosine similarity
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        
        # FIX 3: ALWAYS scale for DINOv3 (don't comment this out!)
        if self.arc == 'dinov3':
            distances = distances * self.dinov3_feature_scale
        
        return distances

    def prototype_distances(self, x):
        """Compute both types of distances"""
        conv_features = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_distances = self._project2basis(conv_features)
        return project_distances, cosine_distances

    def global_min_pooling(self, distances):
        """Global min pooling over spatial dimensions"""
        min_distances = -F.max_pool2d(
            -distances,
            kernel_size=(distances.size()[2], distances.size()[3])
        )
        min_distances = min_distances.view(-1, self.num_prototypes)
        return min_distances

    def global_max_pooling(self, distances):
        """Global max pooling over spatial dimensions"""
        max_distances = F.max_pool2d(
            distances,
            kernel_size=(distances.size()[2], distances.size()[3])
        )
        max_distances = max_distances.view(-1, self.num_prototypes)
        return max_distances

    def subpatch_dist(self, x):
        """
        Compute distances for all subpatches
        Returns:
            conv_feature: [bsz, dim, H, W]
            dist_all: [bsz, num_proto, H*W, n_subpatches]
        """
        dist_all = []
        conv_feature = self.conv_features(x)
        conv_features_normed = F.normalize(conv_feature, p=2, dim=1)
        
        n_p = self.prototype_shape[-1]
        for i in range(n_p):
            # Normalize each subpatch separately
            proto_i = F.normalize(
                self.prototype_vectors[:, :, i], p=2, dim=1
            ).unsqueeze(-1).unsqueeze(-1)
            
            dist_i = F.conv2d(input=conv_features_normed, weight=proto_i)
            
            # FIX 4: Scale here too for consistency
            if self.arc == 'dinov3':
                dist_i = dist_i * self.dinov3_feature_scale
            
            dist_i = dist_i.flatten(2).unsqueeze(-1)  # [bsz, n_proto, H*W, 1]
            dist_all.append(dist_i)
        
        dist_all = torch.cat(dist_all, dim=-1)
        
        return conv_feature, dist_all

    def neigboring_mask(self, center_indices):
        """
        Create neighboring mask for spatial adjacency
        Input: center_indices [bsz, num_prototypes, 1]
        Output: mask [bsz, num_prototypes, num_spatial_patches]
        """
        batch_size, num_points, _ = center_indices.shape
        spatial_size = self.spatial_size
        
        # Padded size
        large_padded_size = spatial_size + self.radius * 2
        large_padded = large_padded_size ** 2
        large_matrix = torch.zeros(batch_size, self.num_prototypes, large_padded).cuda()
        
        small_total = (2 * self.radius + 1) ** 2
        small_matrix = torch.ones(batch_size, self.num_prototypes, small_total).cuda()
        small_size = int(small_matrix.shape[-1] ** 0.5)
        large_size = int(large_matrix.shape[-1] ** 0.5)
        
        # Convert flat indices to 2D coordinates
        center_row = center_indices.squeeze(-1) // spatial_size
        center_col = center_indices.squeeze(-1) % spatial_size
        start_row = torch.tensor(center_row + self.radius - small_size // 2)
        start_col = torch.tensor(center_col + self.radius - small_size // 2)
        start_row = torch.clamp(start_row, 0, large_size - small_size)
        start_col = torch.clamp(start_col, 0, large_size - small_size)
        
        for i in range(small_size):
            for j in range(small_size):
                large_row = start_row + i
                large_col = start_col + j
                large_idx = large_row * large_size + large_col
                small_idx = i * small_size + j
                large_matrix.view(batch_size, num_points, -1)[
                    torch.arange(batch_size)[:, None],
                    torch.arange(num_points),
                    large_idx
                ] += small_matrix[..., small_idx]
        
        large_matrix_reshape = large_matrix.view(batch_size, num_points, large_size, large_size)
        large_matrix_unpad = large_matrix_reshape[
            :, :,
            self.radius:-self.radius,
            self.radius:-self.radius
        ]
        large_matrix_unpad = large_matrix_unpad.reshape(batch_size, num_points, -1)
        
        return large_matrix_unpad

    def greedy_distance(self, x, get_f=False):
        """Greedy prototype selection algorithm"""
        conv_features, dist_all = self.subpatch_dist(x)
        
        slots = torch.sigmoid(self.patch_select * self.temp)
        factor = ((slots.sum(-1))).unsqueeze(-1) + 1e-10
        n_p = self.prototype_shape[-1]
        
        mask_act = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2])).cuda()
        mask_subpatch = torch.ones((x.shape[0], self.num_prototypes, n_p)).cuda()
        mask_all = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2], n_p)).cuda()
        adjacent_mask = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2])).cuda()
        
        indices = torch.FloatTensor().cuda()
        values = torch.FloatTensor().cuda()
        subpatch_ids = torch.LongTensor().cuda()
        
        for _ in range(n_p):
            dist_all_masked = dist_all + (1 - mask_all * adjacent_mask.unsqueeze(-1)) * (-1e5)
            max_subs, max_subs_id = dist_all_masked.max(2)
            max_sub_act, max_sub_act_id = max_subs.max(-1)
            max_patch_id = max_subs_id.gather(-1, max_sub_act_id.unsqueeze(-1))
            
            adjacent_mask = self.neigboring_mask(max_patch_id)
            mask_act = mask_act.scatter(index=max_patch_id, dim=2, value=0)
            mask_subpatch = mask_subpatch.scatter(index=max_sub_act_id.unsqueeze(-1), dim=2, value=0)
            
            mask_all = mask_all * mask_act.unsqueeze(-1)
            mask_all = mask_all.permute(0, 1, 3, 2)
            mask_all = mask_all * mask_subpatch.unsqueeze(-1)
            mask_all = mask_all.permute(0, 1, 3, 2)
            
            max_sub_act = max_sub_act.unsqueeze(-1)
            subpatch_ids = torch.cat([subpatch_ids, max_sub_act_id.unsqueeze(-1)], dim=-1)
            indices = torch.cat([indices, max_patch_id], dim=-1)
            values = torch.cat([values, max_sub_act], dim=-1)
        
        subpatch_ids = subpatch_ids.to(torch.int64)
        _, sub_indexes = subpatch_ids.sort(-1)
        values_reordered = torch.gather(values, -1, sub_indexes)
        indices_reordered = torch.gather(indices, -1, sub_indexes)
        
        values_slot = (values_reordered.clone()) * (slots * n_p / factor)
        max_activation_slots = values_slot.sum(-1)
        
        # FIX 5: Adjust distance computation for scaled similarities
        # Since similarities are now scaled by dinov3_feature_scale,
        # the distance should be: (scale * n_p) - max_activation_slots
        min_distances = (self.dinov3_feature_scale * n_p) - max_activation_slots
        
        if get_f:
            return conv_features, min_distances, indices_reordered
        return max_activation_slots, min_distances, values_reordered

    def push_forward(self, x):
        """Forward pass for prototype pushing"""
        conv_output, min_distances, indices = self.greedy_distance(x, get_f=True)
        return conv_output, min_distances, indices

    def forward(self, x):
        """Main forward pass"""
        max_activation, min_distances, values = self.greedy_distance(x)
        logits = self.last_layer(max_activation)
        return logits, min_distances, values

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """Initialize last layer with correct/incorrect class connections"""
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        
        # FIX 6: Adjust weights for scaled activations
        # With scale=10, activations will be ~10x larger
        # So we should reduce the last layer weights proportionally
        correct_class_connection = 1.0 / self.dinov3_feature_scale
        incorrect_class_connection = incorrect_strength / self.dinov3_feature_scale
        
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _initialize_weights(self):
        """Initialize model weights"""
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def initialize_prototypes_from_batch(self, features, labels):
        """
        Initialize prototypes from actual data features
        Should be called during warmup phase
        
        Args:
            features: [bsz, dim, H, W] - extracted features
            labels: [bsz] - class labels
        """
        if self.prototypes_initialized:
            return
        
        print(f"Initializing prototypes from data batch...")
        
        with torch.no_grad():
            # For each class, sample features to initialize prototypes
            for c in range(self.num_classes):
                class_mask = (labels == c)
                if class_mask.sum() == 0:
                    print(f"  Warning: No samples for class {c} in batch")
                    continue
                
                class_features = features[class_mask]  # [n_samples, dim, H, W]
                n_samples = class_features.shape[0]
                
                # Randomly sample patches from this class
                for p in range(self.num_prototypes_per_class):
                    proto_idx = c * self.num_prototypes_per_class + p
                    
                    # Sample random location from random image
                    sample_idx = torch.randint(0, n_samples, (1,)).item()
                    h = torch.randint(0, class_features.shape[2], (1,)).item()
                    w = torch.randint(0, class_features.shape[3], (1,)).item()
                    
                    # Extract feature and normalize
                    proto_feature = class_features[sample_idx, :, h, w]
                    proto_feature = F.normalize(proto_feature.unsqueeze(0), p=2, dim=0)
                    
                    # Initialize all subpatches with this feature
                    for sp in range(self.prototype_shape[-1]):
                        self.prototype_vectors.data[proto_idx, :, sp] = proto_feature.squeeze()
            
            self.prototypes_initialized = True
            mean_norm = torch.stack([
                self.prototype_vectors[:, :, i].norm(dim=1).mean() 
                for i in range(self.prototype_shape[-1])
            ]).mean()
            print(f"  ✓ Initialized prototypes from data. Mean norm: {mean_norm:.4f}")

    def __repr__(self):
        rep = (
            'PPNetDINOv3Fixed(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {},\n'
            '\tdinov3_feature_scale: {}\n'
            ')'
        )
        return rep.format(
            self.features,
            self.img_size,
            self.prototype_shape,
            self.num_classes,
            self.epsilon,
            self.dinov3_feature_scale
        )
        
def construct_PPNetDINO(base_architecture, pretrained=True, img_size=224,
                prototype_shape=(2000, 192, 1, 1), num_classes=200,
                prototype_activation_function='log',
                sig_temp = 1.0,
                radius = 1,
                add_on_layers_type='bottleneck',
                layers=None):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)

    return PPNetDINO(features=features,
                    img_size=img_size,
                    prototype_shape=prototype_shape,
                    num_classes=num_classes,
                    init_weights=True,
                    prototype_activation_function=prototype_activation_function,
                    radius = radius,
                    sig_temp = sig_temp,
                    add_on_layers_type=add_on_layers_type,
                    layers=layers)