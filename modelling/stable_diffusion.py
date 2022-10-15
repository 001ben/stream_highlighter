!pip install diffusers==0.3.0 transformers scipy ftfy
!conda install pytorch torchvision

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

MY_TOKEN = "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"
from diffusers import StableDiffusionPipeline

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=MY_TOKEN)
pipe = pipe.to(device)

def dummy_checker(images, **kwargs): return images, False

pipe.safety_checker = dummy_checker

prompt = "magician, realistic portrait, symmetrical, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, cinematic lighting, art by artgerm and greg rutkowski and alphonse mucha"
with autocast("cuda"):
    full_return = pipe(prompt, guidance_scale=7.5)

full_return['images'][0]

from PIL import Image
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

num_images = 2
prompt = ["man performing parkour, highly detailed, scenic, digital painting"] * num_images
with autocast("cuda"):
    images = pipe(prompt).images
grid = image_grid(images, rows=1, cols=3)

# import torch
# x = torch.rand(5, 3)
# print(x)
# torch.cuda.is_available()
# torch.zeros(1).cuda()
# torch

