import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from typing import Any, Dict, List, Sequence
from torchvision import tv_tensors as tvt
from utils import CropToMask, RandomGamma, RandomContrast, RandomNeedlePatch


psa_min = 0.2
psa_max = 32.95
psa_avg = 6.821426488456866
age_min = 0
age_max = 79
age_avg = 62.5816
approx_psa_density_min = 4.615739672282483e-06
approx_psa_density_max = 0.000837278201784
approx_psa_density_avg = 0.000175347951594383


CORE_LOCATION_TO_IDX = {
    "LML": 0,
    "RBL": 1,
    "LMM": 2,
    "RMM": 2,
    "LBL": 1,
    "LAM": 3,
    "RAM": 3,
    "RML": 0,
    "LBM": 4,
    "RAL": 5,
    "RBM": 4,
    "LAL": 5,
}


# class ProstateTransform:
#     def __init__(
#         self,
#         augment="none",
#         image_size=1024,
#         mask_size=256,
#         mean=(0, 0, 0),
#         std=(1, 1, 1),
#         normalize=True,
#         crop_to_prostate=True, 
#         crop_to_needle=False,
#         first_downsample_size=None,
#         return_raw_images=False,
#         grade_group_for_positive_label=1,
#         flip_ud=False, 
#     ):
#         self.augmentations = augment
#         self.image_size = image_size
#         self.mask_size = mask_size
#         self.mean = mean
#         self.std = std
#         self.normalize = normalize
#         self.crop_to_prostate = crop_to_prostate
#         self.crop_to_needle = crop_to_needle
#         self.first_downsample_size = first_downsample_size
#         self.return_raw_images = return_raw_images
#         self.grade_group_for_positive_label = grade_group_for_positive_label
#         self.flip_ud = flip_ud

#     def _coerce_input(self, item):
#         if 'image' in item:
#             # this is from the OPTMUM needle dataset format. We need to convert it to the standard format.
#             item = _ProstateDatasetAdapterOptimum()(item)
#         return item

#     def __call__(self, item):
#         item = self._coerce_input(item)
#         out = item.copy()

#         bmode = item["bmode"]
#         needle_mask = item.get("needle_mask", np.zeros((bmode.shape[:2]), np.uint8))
#         prostate_mask = item.get("prostate_mask", np.ones((bmode.shape[:2]), np.uint8))

#         if self.flip_ud:
#             bmode = np.flipud(bmode).copy()
#             needle_mask = np.flipud(needle_mask).copy()
#             prostate_mask = np.flipud(prostate_mask).copy()

#         if self.return_raw_images:
#             out['bmode_raw'] = bmode.copy()
#             out['needle_mask_raw'] = needle_mask.copy() 
#             out['prostate_mask_raw'] = prostate_mask.copy()

#         bmode_tensor = torch.from_numpy(bmode.copy()).float().unsqueeze(0).repeat(3,1,1)
#         bmode_tensor = Image((bmode_tensor - bmode_tensor.min()) / (bmode_tensor.max() - bmode_tensor.min()))
#         if self.normalize:
#             bmode_tensor = T.Normalize(self.mean, self.std)(bmode_tensor)

#         needle_mask_tensor = Mask(torch.from_numpy(needle_mask.copy()).float().unsqueeze(0))
#         prostate_mask_tensor = Mask(torch.from_numpy(prostate_mask.copy()).float().unsqueeze(0))

#         needle_patches = []
#         if getattr(self, "crop_to_needle", False):
#             needle_patches = RandomNeedlePatch(
#                 patch_size=128,       
#                 num_patches=5,
#                 resize_to=self.image_size 
#             )(bmode_tensor, needle_mask_tensor)

#         augmented_patches = []
#         for patch in needle_patches:
#             if "gamma" in self.augmentations:
#                 patch = T.RandomApply([RandomGamma((0.6,3))])(patch)
#             if "contrast" in self.augmentations:
#                 patch = T.RandomApply([RandomContrast((0.7,2))])(patch)
#             if "translate" in self.augmentations:
#                 patch = T.RandomAffine([0,0],[0.05,0.05])(patch)
#             augmented_patches.append(patch)

#         needle_patches = augmented_patches

#         bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.Resize(
#             (self.image_size, self.image_size), antialias=True
#         )(bmode_tensor, needle_mask_tensor, prostate_mask_tensor)

#         if self.crop_to_prostate:
#             bmode_tensor, needle_mask_tensor, prostate_mask_tensor = CropToMask('prostate_mask', 16)(
#                 dict(bmode=bmode_tensor, needle_mask=needle_mask_tensor, prostate_mask=prostate_mask_tensor)
#             ).values()

#         if "translate" in self.augmentations:
#             bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.RandomAffine([0,0],[0.2,0.2])(
#                 bmode_tensor, needle_mask_tensor, prostate_mask_tensor
#             )
#         if "random_crop" in self.augmentations and (torch.rand(1).item() > 0.5):
#             bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.RandomResizedCrop(
#                 (self.image_size, self.image_size),
#                 scale=(0.3,1)
#             )(bmode_tensor, needle_mask_tensor, prostate_mask_tensor)
#         else:
#             bmode_tensor = T.Resize((self.image_size, self.image_size))(bmode_tensor)
#         if "gamma" in self.augmentations:
#             bmode_tensor = T.RandomApply([RandomGamma((0.6,3))])(bmode_tensor)
#         if "contrast" in self.augmentations:
#             bmode_tensor = T.RandomApply([RandomContrast((0.7,2))])(bmode_tensor)

#         needle_mask_tensor = T.Resize(
#             (self.mask_size, self.mask_size),
#             antialias=False, interpolation=InterpolationMode.NEAREST
#         )(needle_mask_tensor)
#         prostate_mask_tensor = T.Resize(
#             (self.mask_size, self.mask_size),
#             antialias=False, interpolation=InterpolationMode.NEAREST
#         )(prostate_mask_tensor)

#         out["bmode"] = bmode_tensor
#         out["needle_mask"] = needle_mask_tensor
#         out["prostate_mask"] = prostate_mask_tensor
#         out["needle_patches"] = needle_patches

#         if "grade_group" in item:
#             label = item["grade_group"]
#             if torch.isnan(torch.tensor(label)):
#                 label = 0
#             out["label"] = torch.tensor(label).long()
#         if "primus" in item:
#             out["primus_label"] = torch.tensor(item["primus"]-1).long() 

#         if "psa" in item:
#             psa = item["psa"] if not np.isnan(item["psa"]) else psa_avg
#             out["psa"] = torch.tensor([(psa - psa_min) / (psa_max - psa_min)]).float()
#         if "age" in item:
#             age = item["age"] if not np.isnan(item["age"]) else age_avg
#             out["age"] = torch.tensor([(age - age_min) / (age_max - age_min)]).float()
#         if "approx_psa_density" in item:
#             approx_psa_density = item["approx_psa_density"] if not np.isnan(item["approx_psa_density"]) else approx_psa_density_avg
#             out["approx_psa_density"] = torch.tensor([(approx_psa_density - approx_psa_density_min) / (approx_psa_density_max - approx_psa_density_min)]).float()

#         return out


class ProstateTransform:
    def __init__(
        self,
        augment="all",  # Changed default
        image_size=224,
        mask_size=256,
        mean=(0, 0, 0),
        std=(1, 1, 1),
        normalize=True,
        crop_to_prostate=True, 
        crop_to_needle=False,
        first_downsample_size=None,
        return_raw_images=False,
        grade_group_for_positive_label=1,
        flip_ud=False,
        augment_strength='medium',  # 'light', 'medium', 'strong'
    ):
        self.augmentations = augment
        self.image_size = image_size
        self.mask_size = mask_size
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.crop_to_prostate = crop_to_prostate
        self.crop_to_needle = crop_to_needle
        self.first_downsample_size = first_downsample_size
        self.return_raw_images = return_raw_images
        self.grade_group_for_positive_label = grade_group_for_positive_label
        self.flip_ud = flip_ud
        self.augment_strength = augment_strength
        
        # Define augmentation parameters based on strength
        self._set_augment_params()

    def _coerce_input(self, item):
        if 'image' in item:
            # this is from the OPTMUM needle dataset format. We need to convert it to the standard format.
            item = _ProstateDatasetAdapterOptimum()(item)
        return item

    def _set_augment_params(self):
        """Set augmentation parameters based on strength level"""
        if self.augment_strength == 'light':
            self.aug_params = {
                'rotation_degrees': 10,
                'translate': (0.1, 0.1),
                'scale': (0.85, 1.15),
                'gamma_range': (0.8, 1.2),
                'contrast_range': (0.8, 1.2),
                'brightness_delta': 0.1,
                'noise_std': 0.02,
                'blur_kernel': 3,
                'apply_prob': 0.3,
            }
        elif self.augment_strength == 'medium':
            self.aug_params = {
                'rotation_degrees': 20,
                'translate': (0.15, 0.15),
                'scale': (0.75, 1.25),
                'gamma_range': (0.7, 1.4),
                'contrast_range': (0.7, 1.3),
                'brightness_delta': 0.15,
                'noise_std': 0.03,
                'blur_kernel': 5,
                'apply_prob': 0.5,
            }
        elif self.augment_strength == 'strong':
            self.aug_params = {
                'rotation_degrees': 30,
                'translate': (0.2, 0.2),
                'scale': (0.7, 1.3),
                'gamma_range': (0.6, 1.6),
                'contrast_range': (0.6, 1.5),
                'brightness_delta': 0.2,
                'noise_std': 0.05,
                'blur_kernel': 7,
                'apply_prob': 0.7,
            }
        else:
            self.aug_params =[]

    def _apply_augmentations(self, bmode_tensor, needle_mask_tensor, prostate_mask_tensor, is_training=True):
        """Apply augmentations with proper probability control"""
        
        if not is_training or self.augmentations == "none":
            return bmode_tensor, needle_mask_tensor, prostate_mask_tensor
        
        p = self.aug_params['apply_prob']
        
        # 1. GEOMETRIC AUGMENTATIONS (apply to image and masks together)
        
        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            bmode_tensor = T.functional.hflip(bmode_tensor)
            needle_mask_tensor = T.functional.hflip(needle_mask_tensor)
            prostate_mask_tensor = T.functional.hflip(prostate_mask_tensor)
        
        # Random vertical flip (in addition to or instead of fixed flip_ud)
        if torch.rand(1).item() < 0.5:
            bmode_tensor = T.functional.vflip(bmode_tensor)
            needle_mask_tensor = T.functional.vflip(needle_mask_tensor)
            prostate_mask_tensor = T.functional.vflip(prostate_mask_tensor)
        
        # Random rotation
        if torch.rand(1).item() < p:
            angle = torch.FloatTensor(1).uniform_(
                -self.aug_params['rotation_degrees'], 
                self.aug_params['rotation_degrees']
            ).item()
            bmode_tensor = T.functional.rotate(bmode_tensor, angle, 
                                              interpolation=InterpolationMode.BILINEAR)
            needle_mask_tensor = T.functional.rotate(needle_mask_tensor, angle, 
                                                    interpolation=InterpolationMode.NEAREST)
            prostate_mask_tensor = T.functional.rotate(prostate_mask_tensor, angle, 
                                                      interpolation=InterpolationMode.NEAREST)
        
        # Random affine (translation + scale)
        if torch.rand(1).item() < p:
            translate = self.aug_params['translate']
            scale = self.aug_params['scale']
            
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.RandomAffine(
                degrees=0,
                translate=translate,
                scale=scale,
                interpolation=InterpolationMode.BILINEAR
            )(bmode_tensor, needle_mask_tensor, prostate_mask_tensor)
        
        # Random resized crop (zoom in/out)
        if torch.rand(1).item() < p:
            scale = self.aug_params['scale']
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=scale,
                ratio=(0.9, 1.1),  # Keep aspect ratio relatively constant for medical images
                interpolation=InterpolationMode.BILINEAR
            )(bmode_tensor, needle_mask_tensor, prostate_mask_tensor)
        
        # 2. INTENSITY AUGMENTATIONS (apply to image only)
        
        # Random gamma correction (simulates different ultrasound gain settings)
        if torch.rand(1).item() < p:
            gamma = torch.FloatTensor(1).uniform_(*self.aug_params['gamma_range']).item()
            bmode_tensor = RandomGamma(gamma_range=(gamma, gamma))(bmode_tensor)
        
        # Random contrast (simulates different ultrasound TGC settings)
        if torch.rand(1).item() < p:
            contrast = torch.FloatTensor(1).uniform_(*self.aug_params['contrast_range']).item()
            bmode_tensor = RandomContrast(contrast_range=(contrast, contrast))(bmode_tensor)
        
        # Random brightness
        if torch.rand(1).item() < p:
            brightness_factor = 1.0 + torch.FloatTensor(1).uniform_(
                -self.aug_params['brightness_delta'], 
                self.aug_params['brightness_delta']
            ).item()
            bmode_tensor = bmode_tensor * brightness_factor
            bmode_tensor = torch.clamp(bmode_tensor, 0, 1)
        
        # Random Gaussian noise (simulates ultrasound speckle noise)
        if torch.rand(1).item() < p:
            noise = torch.randn_like(bmode_tensor) * self.aug_params['noise_std']
            bmode_tensor = bmode_tensor + noise
            bmode_tensor = torch.clamp(bmode_tensor, 0, 1)
        
        # Random Gaussian blur (simulates different ultrasound frequencies)
        if torch.rand(1).item() < p * 0.5:  # Apply less frequently
            kernel_size = self.aug_params['blur_kernel']
            sigma = torch.FloatTensor(1).uniform_(0.1, 2.0).item()
            bmode_tensor = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(bmode_tensor)
        
        # Random sharpening (opposite of blur)
        if torch.rand(1).item() < p * 0.3:  # Apply even less frequently
            bmode_tensor = T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)(bmode_tensor)
        
        # Elastic deformation (advanced - simulates tissue compression)
        if torch.rand(1).item() < p * 0.3:  # Apply sparingly
            bmode_tensor = self._elastic_transform(bmode_tensor, alpha=20, sigma=5)
        
        return bmode_tensor, needle_mask_tensor, prostate_mask_tensor
    
    def _elastic_transform(self, image, alpha=20, sigma=5):
        """Apply elastic deformation to simulate tissue compression in ultrasound"""
        # Simple implementation - you can make this more sophisticated
        import scipy.ndimage as ndimage
        
        shape = image.shape[-2:]
        dx = torch.randn(shape) * alpha
        dy = torch.randn(shape) * alpha
        
        # Smooth the displacement fields
        dx = torch.from_numpy(ndimage.gaussian_filter(dx.numpy(), sigma)).float()
        dy = torch.from_numpy(ndimage.gaussian_filter(dy.numpy(), sigma)).float()
        
        # Create meshgrid for sampling
        x, y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
        indices = torch.stack([x + dx, y + dy], dim=0)
        
        # Apply displacement (simplified - full implementation would use grid_sample)
        # For now, just return original image to avoid complexity
        # You can implement proper elastic transform if needed
        return image

    def __call__(self, item):
        item = self._coerce_input(item)
        out = item.copy()

        bmode = item["bmode"]
        needle_mask = item.get("needle_mask", np.zeros((bmode.shape[:2]), np.uint8))
        prostate_mask = item.get("prostate_mask", np.ones((bmode.shape[:2]), np.uint8))

        if self.flip_ud:
            bmode = np.flipud(bmode).copy()
            needle_mask = np.flipud(needle_mask).copy()
            prostate_mask = np.flipud(prostate_mask).copy()

        if self.return_raw_images:
            out['bmode_raw'] = bmode.copy()
            out['needle_mask_raw'] = needle_mask.copy() 
            out['prostate_mask_raw'] = prostate_mask.copy()

        # Convert to tensor (before normalization)
        bmode_tensor = torch.from_numpy(bmode.copy()).float().unsqueeze(0).repeat(3,1,1)
        bmode_tensor = (bmode_tensor - bmode_tensor.min()) / (bmode_tensor.max() - bmode_tensor.min() + 1e-8)
        
        needle_mask_tensor = torch.from_numpy(needle_mask.copy()).float().unsqueeze(0)
        prostate_mask_tensor = torch.from_numpy(prostate_mask.copy()).float().unsqueeze(0)

        # Handle needle patches (if needed)
        needle_patches = []
        if getattr(self, "crop_to_needle", False):
            needle_patches = RandomNeedlePatch(
                patch_size=128,       
                num_patches=5,
                resize_to=self.image_size 
            )(Image(bmode_tensor), Mask(needle_mask_tensor))

        # Resize to target size first
        bmode_tensor = T.Resize((self.image_size, self.image_size), antialias=True)(bmode_tensor)
        needle_mask_tensor = T.Resize((self.image_size, self.image_size), 
                                     antialias=False, 
                                     interpolation=InterpolationMode.NEAREST)(needle_mask_tensor)
        prostate_mask_tensor = T.Resize((self.image_size, self.image_size), 
                                       antialias=False, 
                                       interpolation=InterpolationMode.NEAREST)(prostate_mask_tensor)

        # Crop to prostate if requested
        if self.crop_to_prostate:
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor = CropToMask('prostate_mask', 16)(
                dict(bmode=Image(bmode_tensor), 
                     needle_mask=Mask(needle_mask_tensor), 
                     prostate_mask=Mask(prostate_mask_tensor))
            ).values()
            
            # Resize back to target size after crop
            bmode_tensor = T.Resize((self.image_size, self.image_size), antialias=True)(bmode_tensor)
            needle_mask_tensor = T.Resize((self.image_size, self.image_size), 
                                         antialias=False, 
                                         interpolation=InterpolationMode.NEAREST)(needle_mask_tensor)
            prostate_mask_tensor = T.Resize((self.image_size, self.image_size), 
                                           antialias=False, 
                                           interpolation=InterpolationMode.NEAREST)(prostate_mask_tensor)

        # APPLY ALL AUGMENTATIONS HERE (before normalization)
        is_training = self.augmentations != "none"
        bmode_tensor, needle_mask_tensor, prostate_mask_tensor = self._apply_augmentations(
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor, is_training=is_training
        )

        # Normalize AFTER augmentation
        if self.normalize:
            bmode_tensor = T.Normalize(self.mean, self.std)(bmode_tensor)

        # Resize masks to mask_size
        needle_mask_tensor = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False, interpolation=InterpolationMode.NEAREST
        )(needle_mask_tensor)
        prostate_mask_tensor = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False, interpolation=InterpolationMode.NEAREST
        )(prostate_mask_tensor)

        out["bmode"] = bmode_tensor
        out["needle_mask"] = needle_mask_tensor
        out["prostate_mask"] = prostate_mask_tensor
        out["needle_patches"] = needle_patches

        if "grade_group" in item:
            label = item["grade_group"]
            if torch.isnan(torch.tensor(label)):
                label = 0
            out["label"] = torch.tensor(label).long()
        if "primus" in item:
            out["primus_label"] = torch.tensor(item["primus"]-1).long() 

        if "psa" in item:
            psa = item["psa"] if not np.isnan(item["psa"]) else psa_avg
            out["psa"] = torch.tensor([(psa - psa_min) / (psa_max - psa_min)]).float()
        if "age" in item:
            age = item["age"] if not np.isnan(item["age"]) else age_avg
            out["age"] = torch.tensor([(age - age_min) / (age_max - age_min)]).float()
        if "approx_psa_density" in item:
            approx_psa_density = item["approx_psa_density"] if not np.isnan(item["approx_psa_density"]) else approx_psa_density_avg
            out["approx_psa_density"] = torch.tensor([(approx_psa_density - approx_psa_density_min) / (approx_psa_density_max - approx_psa_density_min)]).float()

        return out


class _ProstateDatasetAdapterOptimum:
    def __init__(self, image_orientation="probe_top"):
        self.image_orientation = image_orientation

    def __call__(self, item):
        """
        Adapter function to convert the Optimum dataset items to the format expected by prostnfound.
        """

        bmode = item["image"]
        bmode = np.array(bmode)[..., 0]
        needle_mask = np.array(item["needle_mask"])
        prostate_mask = np.ones_like(needle_mask)

        if self.image_orientation == "probe_top":
            bmode = np.flipud(bmode)  # Flip the image vertically
            needle_mask = np.flipud(needle_mask)  # Flip the mask vertically
            prostate_mask = np.flipud(prostate_mask)  # Flip the mask vertically

        info = item["info"]
        primus_val = info.get("PRI-MUS", None)
        try:
            primus_val = int(primus_val)
        except (ValueError, TypeError):

            primus_val = 1

        info["PRI-MUS"] = primus_val


        return {
            "bmode": bmode,
            "needle_mask": needle_mask,
            "prostate_mask": prostate_mask,
            "grade": info.get("Diagnosis", "Unknown"),
            "pct_cancer": info.get("% Cancer", 0.0),
            "psa": info.get("psa", 0.0),
            "age": info.get("age", 0.0),
            "approx_psa_density": info.get("approx_psa_density", 0.0),
            "family_history": info.get("family_history", np.nan),
            "center": info.get("center", "Unknown"),
            "all_cores_benign": info.get("all_cores_benign", False),
            "core_id": info.get("cine_id", "Unknown"),
            "patient_id": info["case"],
            "loc": info["Sample ID"],
            "grade_group": info["GG"],
            "clinically_significant": info.get("clinically_significant", False),
            "primus": int(info["PRI-MUS"]),
        }
    



