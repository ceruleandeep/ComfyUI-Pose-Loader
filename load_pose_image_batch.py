"""
@author: ceruleandeep
@title: Pose Image Loader
@nickname: Pose Image Loader
@description: This extension provides a node to load OpenPose annotation images.

From: ComfyUI-Inspire-Pack https://github.com/ltdrdata/ComfyUI-Inspire-Pack
"""

import io
import logging
import os
import time
import zipfile
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Generator, Tuple

import PIL
import comfy
import folder_paths
import numpy as np
import torch
from PIL import Image
from PIL import ImageOps

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# logger.addHandler(logging.StreamHandler())


class PLFilter(Enum):
    ALL = "All"
    POSES = "OpenPoses"
    DEPTH = "Depth maps"


class PLLoadImagesFromDirBatch:
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".zip")

    # noinspection PyPep8Naming
    # noinspection PyMethodParameters
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "directory": (s.pose_dirs(),),
                "image_filter": (
                    tuple(e.value for e in PLFilter),
                    {"default": PLFilter.ALL.value, "label": "Filter"},
                ),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "recursive": (
                    "BOOLEAN",
                    {"default": False, "label_on": "yes", "label_off": "no"},
                ),
            },
        }

        return inputs

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "mask", "batch_size", "width", "height")
    FUNCTION = "load_images"
    CATEGORY = "image"

    def load_images(
        self,
        directory: str,
        image_filter: str = PLFilter.ALL.value,
        image_load_cap: int = 0,
        start_index: int = 0,
        recursive: bool = False,
    ):
        if directory == "(all)":
            directory = ""

        directory = self.pose_path() / directory
        image_load_cap = max(0, image_load_cap) if image_load_cap else 0
        start_index = max(0, start_index) if start_index else 0
        end_index = start_index + image_load_cap if image_load_cap else None

        filt = PLFilter(image_filter) if isinstance(image_filter, str) else PLFilter.ALL

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' not found")

        if not (dir_files := self._imgs_and_zipfiles_in_dir(directory, recursive)):
            subdirs_str = " or subdirectories " if recursive else ""
            raise FileNotFoundError(f"No image files in '{directory}'{subdirs_str}")

        # image filtering is in here
        pil_images = self._pil_images_for_files(dir_files, filt, image_load_cap)
        pil_image_slice = islice(pil_images, start_index, end_index)

        images_and_masks = self._img_and_mask_tensors(pil_image_slice)

        try:
            img_batch, mask = next(images_and_masks)
        except StopIteration:
            msg = f"No images in '{directory}' after filtering (start {start_index}, cap {image_load_cap}, filter: {filt.value})"
            logger.warning(msg)
            raise FileNotFoundError(msg)

        # I don't think ComfyUI-Inspire-Pack did anything special with the masks
        # when making the batch, so I'm not going to either
        masks = [mask]
        x, y = img_batch.shape[1:3]
        for image2, mask2 in images_and_masks:
            if img_batch.shape[1:] != image2.shape[1:]:
                image2 = comfy.utils.common_upscale(
                    image2.movedim(-1, 1),
                    img_batch.shape[2],
                    img_batch.shape[1],
                    "bilinear",
                    "center",
                ).movedim(1, -1)
            img_batch = torch.cat((img_batch, image2), dim=0)
            masks.append(mask2)

        return img_batch, masks, len(masks), x, y

    def _img_and_mask_tensors(self, img_slice) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Convert a list of PIL images to a list of Torch tensors.
        """

        for i in img_slice:
            image, mask = self._img_and_mask_tensor(i)
            yield image, mask

    def _pil_images_for_files(self, dir_files: list[str], filt=PLFilter.ALL, image_load_cap=0) -> list:
        """
        Load PIL images from a list of files including within zip files.
        """

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue

            if zipfile.is_zipfile(image_path):
                for i in self._pil_images_from_zip(image_path):
                    if self._matches_filter(i, filt):
                        yield i

            else:
                try:
                    i = Image.open(image_path)
                except PIL.UnidentifiedImageError:
                    continue
                if self._matches_filter(i, filt):
                    i = ImageOps.exif_transpose(i)
                    yield i

    def _pil_images_from_zip(self, image_path: str):
        with zipfile.ZipFile(image_path, "r") as zip_ref:
            for i, file_info in enumerate(zip_ref.infolist()):
                if file_info.filename.lower().endswith(self.valid_extensions):
                    with zip_ref.open(file_info) as file:
                        pil_image = Image.open(io.BytesIO(file.read()))
                        pil_image = ImageOps.exif_transpose(pil_image)
                        yield pil_image

    def _img_and_mask_tensor(self, pil_image: Image.Image):
        """
        Convert a PIL image to a torch tensor.
        """

        image = pil_image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in pil_image.getbands():
            mask = np.array(pil_image.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return image, mask

    def _imgs_and_zipfiles_in_dir(self, directory, recursive=False) -> list[str]:
        dir_files = []
        for ext in self.valid_extensions:
            if recursive:
                dir_files.extend(Path(directory).rglob(f"*{ext}"))
            else:
                dir_files.extend(Path(directory).glob(f"*{ext}"))
        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]
        return dir_files

    @classmethod
    def _is_openpose_annotation(cls, image):
        """
        Check if the image is probably an OpenPose annotation.
        Quick and dirty check: the top-left pixel should be black.
        """

        if image.mode in ["L", "LA", "P", "PA"]:
            # not a color image
            return False

        top_left_pixel = image.getpixel((0, 0))

        # If it's not a pixel tuple, it's not an OpenPose image
        if not isinstance(top_left_pixel, tuple):
            return False

        # we can throw out some depth maps here
        if len(top_left_pixel) == 4 and top_left_pixel[3] != 255:
            return False

        if top_left_pixel[:3] != (0, 0, 0):
            return False

        # If it's a depth map, it's not an OpenPose annotation
        # but finding out is too slow
        # if cls._is_depth_map(image):
        #     return False

        return True

    @staticmethod
    def _is_depth_map(image):
        """
        Check if the image is probably a depth map.
        """

        if image.mode in ["L", "LA"]:
            # Canny?
            return False

        if image.mode in ["P"]:
            return True

        image_array = np.array(image)

        if len(image_array.shape) == 2:
            # Canny again?
            return False

        if image_array.shape[2] < 3:
            # Canny with alpha?
            return False

        # this is slow, but ways to speed it up are also slow
        try:
            b, g, r = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
        except IndexError:
            logger.debug(f"    {image} {image_array.shape=} IE")
            return True
        grayscale = (b == g).all() and (b == r).all()

        return grayscale

    @classmethod
    def pose_path(cls) -> Path:
        """
        Return the path to the pose packs. If the user has set a custom path, use that.
        If multiple paths are set, use the first one.
        """

        if paths_and_extns := folder_paths.folder_names_and_paths.get("poses", None):
            paths, extns = paths_and_extns
            return Path(paths[0])
        return Path(__file__).parent / "models" / "poses"

    @classmethod
    def pose_dirs(cls):
        """
        Return a list of pose pack subdirectories.
        """

        pose_path = cls.pose_path()
        if not pose_path.is_dir():
            raise FileNotFoundError(f"Pose pack directory '{pose_path}' not found")

        subdirs = ["(all)"] + sorted([str(p.relative_to(pose_path)) for p in pose_path.glob("*") if p.is_dir()])
        return subdirs

    def _matches_filter(self, i, filt):
        if filt == PLFilter.POSES:
            return self._is_openpose_annotation(i)

        if filt == PLFilter.DEPTH:
            return self._is_depth_map(i)

        return True


NODE_CLASS_MAPPINGS = {
    "PLLoadImagesFromDirBatch": PLLoadImagesFromDirBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PLLoadImagesFromDirBatch": "Load Image Batch From Dir (Pose Loader)",
}
