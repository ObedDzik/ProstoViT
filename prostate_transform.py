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


class ProstateTransform:
    def __init__(
        self,
        augment="none",
        image_size=1024,
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

    def _coerce_input(self, item):
        if 'image' in item:
            # this is from the OPTMUM needle dataset format. We need to convert it to the standard format.
            item = _ProstateDatasetAdapterOptimum()(item)
        return item

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

        bmode_tensor = torch.from_numpy(bmode.copy()).float().unsqueeze(0).repeat(3,1,1)
        bmode_tensor = Image((bmode_tensor - bmode_tensor.min()) / (bmode_tensor.max() - bmode_tensor.min()))
        if self.normalize:
            bmode_tensor = T.Normalize(self.mean, self.std)(bmode_tensor)

        needle_mask_tensor = Mask(torch.from_numpy(needle_mask.copy()).float().unsqueeze(0))
        prostate_mask_tensor = Mask(torch.from_numpy(prostate_mask.copy()).float().unsqueeze(0))

        needle_patches = []
        if getattr(self, "crop_to_needle", False):
            needle_patches = RandomNeedlePatch(
                patch_size=128,       
                num_patches=5,
                resize_to=self.image_size 
            )(bmode_tensor, needle_mask_tensor)

        augmented_patches = []
        for patch in needle_patches:
            if "gamma" in self.augmentations:
                patch = T.RandomApply([RandomGamma((0.6,3))])(patch)
            if "contrast" in self.augmentations:
                patch = T.RandomApply([RandomContrast((0.7,2))])(patch)
            if "translate" in self.augmentations:
                patch = T.RandomAffine([0,0],[0.05,0.05])(patch)
            augmented_patches.append(patch)

        needle_patches = augmented_patches

        bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.Resize(
            (self.image_size, self.image_size), antialias=True
        )(bmode_tensor, needle_mask_tensor, prostate_mask_tensor)

        if self.crop_to_prostate:
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor = CropToMask('prostate_mask', 16)(
                dict(bmode=bmode_tensor, needle_mask=needle_mask_tensor, prostate_mask=prostate_mask_tensor)
            ).values()

        if "translate" in self.augmentations:
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.RandomAffine([0,0],[0.2,0.2])(
                bmode_tensor, needle_mask_tensor, prostate_mask_tensor
            )
        if "random_crop" in self.augmentations and (torch.rand(1).item() > 0.5):
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.RandomResizedCrop(
                (self.image_size, self.image_size),
                scale=(0.3,1)
            )(bmode_tensor, needle_mask_tensor, prostate_mask_tensor)
        else:
            bmode_tensor = T.Resize((self.image_size, self.image_size))(bmode_tensor)
        if "gamma" in self.augmentations:
            bmode_tensor = T.RandomApply([RandomGamma((0.6,3))])(bmode_tensor)
        if "contrast" in self.augmentations:
            bmode_tensor = T.RandomApply([RandomContrast((0.7,2))])(bmode_tensor)

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
    



