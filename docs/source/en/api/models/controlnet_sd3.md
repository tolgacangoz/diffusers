<!--Copyright 2025 The HuggingFace Team and The InstantX Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SD3ControlNetModel

SD3ControlNetModel is an implementation of ControlNet for Stable Diffusion 3.

The ControlNet model was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, Maneesh Agrawala. It provides a greater degree of control over text-to-image generation by conditioning the model on additional inputs such as edge maps, depth maps, segmentation maps, and keypoints for pose detection.

The abstract from the paper is:

*We present ControlNet, a neural network architecture to add spatial conditioning controls to large, pretrained text-to-image diffusion models. ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls. The neural architecture is connected with "zero convolutions" (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning. We test various conditioning controls, eg, edges, depth, segmentation, human pose, etc, with Stable Diffusion, using single or multiple conditions, with or without prompts. We show that the training of ControlNets is robust with small (<50k) and large (>1m) datasets. Extensive results show that ControlNet may facilitate wider applications to control image diffusion models.*

## Loading from the original format

By default the [`SD3ControlNetModel`] should be loaded with [`~ModelMixin.from_pretrained`].

```py
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel

controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
pipe = StableDiffusion3ControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet)
```

## SD3ControlNetModel

[[autodoc]] SD3ControlNetModel

## SD3ControlNetOutput

[[autodoc]] models.controlnets.controlnet_sd3.SD3ControlNetOutput

