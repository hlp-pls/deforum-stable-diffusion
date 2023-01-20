#@title Pixel Shader Animation Definitions

from einops import rearrange, repeat
import cv2
import numpy as np
from PIL import Image

#https://github.com/deforum/stable-diffusion/blob/main/helpers/depth.py
depth_min = 1000
depth_max = -1000

def depth_tensor_to_PIL(depth):
  global depth_min
  global depth_max
  depth = depth.cpu().numpy()
  if len(depth.shape) == 2:
    depth = np.expand_dims(depth, axis=0)
  depth_min = min(depth_min, depth.min())
  depth_max = max(depth_max, depth.max())
  #print(f"  depth min:{depth.min()} max:{depth.max()}")
  denom = max(1e-8, depth_max - depth_min)
  temp = rearrange((depth - depth_min) / denom * 255, 'c h w -> h w c')
  temp = repeat(temp, 'h w 1 -> h w c', c=3)
  return Image.fromarray(temp.astype(np.uint8))

def mgl_render_frame(MGL_fbo, args, anim_args, return_img):
  MGL_fbo.render()

  if return_img == True:
    #https://stackoverflow.com/a/57017295
    img_buf = MGL_fbo.fbo.read(components=4) 
    img = np.frombuffer(img_buf, np.uint8).reshape(MGL_fbo.size[1], MGL_fbo.size[0], 4)[::-1]
    return img
    
def anim_frame_warp_mgl(MGL_fbo, prev_img_cv2, depth_tensor, args, anim_args, keys, frame_idx):
  #if MGL_fbo.depth_set == False:
  depth_pil_image = depth_tensor_to_PIL(depth_tensor)
  MGL_fbo.set_depth_img(depth_pil_image)

  #https://www.zinnunkebi.com/python-opencv-pil-convert/
  color_coverted = cv2.cvtColor(prev_img_cv2, cv2.COLOR_BGR2RGB)
  #https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
  #--> this was causing problems with turbo (cadence)
  pil_image=Image.fromarray((color_coverted).astype(np.uint8))
  MGL_fbo.set_init_img(PILimg=pil_image)
  
  frame_length = anim_args.mgl_steps
  for frame_time in range(frame_length):
    if frame_time < frame_length - 1:
      mgl_render_frame(MGL_fbo, args, anim_args, False)
    else:
      output_pil_img = mgl_render_frame(MGL_fbo, args, anim_args, True)

  output_np_img = np.array(output_pil_img)
  output_img = cv2.cvtColor(output_np_img, cv2.COLOR_RGB2BGR)
  #output_img = sample_to_cv2(output_img, type=np.float32)  
  return output_img
