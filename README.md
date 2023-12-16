# ComfyUI-Pose-Loader

Loads pose packs from a directory and provides them as an image batch. 
Allows you to batch-generate images from pose packs.

Killer feature: you don't have to unzip the pose packs, just put them in a directory and go.
Or let your CivitAI downloader put them there automatically.

Other reasons that this exists:

* reads OpenPose-annotated images and depth maps directly from zip files
* filters out preview images and other junk, so your ControlNet only sees the poses 
* turns your collection of pose packs and misc unsorted OpenPose pngs into a single source of poses
* basically makes it easy to use your giant pile of poses with ComfyUI

![image](https://github.com/ceruleandeep/ComfyUI-Pose-Loader/assets/83318388/13cae6f6-e2ec-4823-824f-38eb1574a2a1)

## Installation

```cd ComfyUI/custom_nodes && git clone https://github.com/ceruleandeep/ComfyUI-Pose-Loader.git```

## Usage

Download pose packs from [CivitAI](https://civitai.com/models?type=poses) or whatever and put them in a directory. 
By default this is `ComfyUI/custom_nodes/ComfyUI-Pose-Loader/models/poses`.

Add a `PLLoadImagesFromDirBatch` node to your graph. Connect the `images` output to a `ControlNetApply` node.

Set the following parameters to your liking:

* `directory`: Subdirectory of your poses directory to load from, or empty for the base directory.
* `image_filter`: Attempts to load only OpenPose annotations or depth maps, so the right poses are sent to the ControlNet.
* `image_load_cap`: Maximum number of images to load from the directory. May return less!
* `start_index`: Index of the first image to load. Useful for loading bits of a pose pack, or when loading multiple pose packs at once.
* `recursive`: Whether to descend into subdirectories of the selected directory.

## Configuration

You probably have a different directory for your pose packs. Configure it in `extra_model_paths.yaml`.

Example configuration to load poses from AUTOMATIC1111:

```yaml
a111:
    base_path: /home/user/stable-diffusion-webui/models/
    poses: Poses
```

## Returns

* `images`: Batch of images, shaped according to the size of the first image loaded.
* `masks`: Batch of masks, shaped however the masks are shaped.
* `count`: Number of images loaded. Feed this into your Efficient Loader or whatever, so you can e.g. generate one image per pose.
* `width`: Width of the images. Theoretically you could use this to generate images with the same size as the pose maps.
* `height`: Height of the images.

## Credits

Batching code adapted from [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)