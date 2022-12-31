#@title Pixel Shader Animation Definitions

def mgl_render_frame(args, anim_args, return_img):
  MGL_fbo.render()

  if return_img == True:
    #https://stackoverflow.com/a/57017295
    img_buf = MGL_fbo.fbo.read(components=4) 
    img = np.frombuffer(img_buf, np.uint8).reshape(MGL_fbo.size[1], MGL_fbo.size[0], 4)[::-1]
    return img
    
def anim_frame_warp_mgl(prev_img_cv2, depth_tensor, args, anim_args, keys, frame_idx):
  if MGL_fbo.depth_set == False:
    depth_pil_image = depth_tensor_to_PIL(depth_tensor)
    MGL_fbo.set_depth_img(depth_pil_image)

  #https://www.zinnunkebi.com/python-opencv-pil-convert/
  color_coverted = cv2.cvtColor(prev_img_cv2, cv2.COLOR_BGR2RGB)
  #https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
  #--> this was causing problems with turbo (cadence)
  pil_image=Image.fromarray((color_coverted).astype(np.uint8))
  MGL_fbo.set_init_img(PILimg=pil_image)
  
  frame_length = anim_args.pixel_shader_steps
  for frame_time in range(frame_length):
    if frame_time < frame_length - 1:
      mgl_render_frame(args, anim_args, False)
    else:
      output_pil_img = mgl_render_frame(args, anim_args, True)

  output_np_img = np.array(output_pil_img)
  output_img = cv2.cvtColor(output_np_img, cv2.COLOR_RGB2BGR)
  #output_img = sample_to_cv2(output_img, type=np.float32)  
  return output_img
