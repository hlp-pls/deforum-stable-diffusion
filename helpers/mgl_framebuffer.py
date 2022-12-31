#@title ModernGL Dependencies (For Pixel Shader Animations)

#https://github.com/moderngl/moderngl/blob/master/examples/headless_egl.py
#https://github.com/moderngl/moderngl/blob/master/examples/raymarching.py

#https://kkiho.tistory.com/16
#https://github.com/moderngl/moderngl/blob/master/examples/basic_uniforms_and_attributes.py

#https://github.com/moderngl/moderngl/blob/master/examples/crate.py

#https://stackoverflow.com/questions/56980266/how-do-i-read-a-moderngl-fboframe-buffer-object-back-into-a-numpy-array

import numpy as np
from PIL import Image
import moderngl

#--> WARNING : when uniforms are unused inside the shader program, an error occurs
#https://github.com/moderngl/moderngl/issues/149  
class FakeUniform:
    value = None

class mglFBO:

  quad_vertex_shader = '''
  #version 330
  in vec2 in_vert;
  in vec2 in_uv;
  out vec2 UV;
  void main() {
      gl_Position = vec4(in_vert, 0.0, 1.0);
      UV = in_uv;
  }
  '''
  copier_fragment_shader = '''
  #version 330
  out vec4 outputColor;
  in vec2 UV;

  uniform sampler2D copy_target;

  void main()
  {
      vec4 copy = texture2D(copy_target,UV);
      outputColor = copy;
  }
  '''

  def __init__(self,**kwargs):
    self.kwargs = kwargs

    self.ctx = kwargs["ctx"]
    self.size = kwargs["size"]
    self.enable_backbuffer = kwargs["enable_backbuffer"]
    
    self.texture = self.ctx.texture(self.size, components=4, dtype="f4")
    self.fbo = self.ctx.framebuffer(self.texture)
    self.prog = self.ctx.program(vertex_shader=self.quad_vertex_shader,fragment_shader=kwargs["fragment_shader"])
    

    vertex_data = np.array([
        # x,    y,   z,    u,   v
        -1.0, -1.0, 0.0,  0.0, 0.0,
        +1.0, -1.0, 0.0,  1.0, 0.0,
        -1.0, +1.0, 0.0,  0.0, 1.0,
        +1.0, +1.0, 0.0,  1.0, 1.0,
    ]).astype(np.float32)

    content = [(
        self.ctx.buffer(vertex_data),
        '3f 2f',
        'in_vert', 'in_uv'
    )]

    self.vao = self.ctx.vertex_array(self.prog, content)

    #https://github.com/moderngl/moderngl/issues/459
    #https://realpython.com/python-keyerror/
    self.time = self.prog.get('time', FakeUniform())
    self.init = self.prog.get('init', FakeUniform())
    self.resolution = self.prog.get('resolution', FakeUniform())

    self.time.value = 0.0
    self.resolution.value = self.size

    if self.enable_backbuffer == True:
      self.bck_texture = self.ctx.texture(self.size, components=4, dtype="f4")
      self.bck_fbo = self.ctx.framebuffer(self.bck_texture)
      self.bck_prog = self.ctx.program(vertex_shader=self.quad_vertex_shader, fragment_shader=self.copier_fragment_shader)
      self.bck_vao = self.ctx.vertex_array(self.bck_prog, content)

      self.backbuffer_uniform = self.prog.get('backbuffer', FakeUniform())
      self.copy_target_uniform = self.bck_prog.get('copy_target', FakeUniform())
    
    self.textures = {}
    self.texture_buffers = {}
    self.texture_uniforms = {}
    self.init_texture = None

    self.depth_set = False

  
  def set_uniform(self, name, value):
    uniform = self.prog.get(name, FakeUniform())
    uniform.value = value
  
  def set_texture_uniform(self, texture, name, id):
    if name not in self.textures:
      self.texture_uniforms[name] = self.prog.get(name, FakeUniform())
      self.textures[name] = texture
    self.texture_uniforms[name].value = id

  

  #https://stackoverflow.com/questions/64074990/using-pillow-image-tobytes-to-flip-the-image-and-swap-the-color-channels
  def set_init_img(self, **kwargs):
    if 'path' in kwargs:
      im = Image.open(kwargs['path'])
    elif 'PILimg' in kwargs:
      im = kwargs['PILimg']
    else:
      print('something will go wrong with init image')

    im_WIDTH, im_HEIGHT, im_DATA = im.size[0], im.size[1], im.convert('RGBA').tobytes("raw", "RGBA", 0, -1)
    self.init_texture = self.ctx.texture((im.size[0], im.size[1]), components=4, data=im_DATA)
    self.init_texture_uniform = self.prog.get('init_texture', FakeUniform())
    self.init.value = 1.0
  
  def set_depth_img(self, pil_img):
    im = pil_img
    im_WIDTH, im_HEIGHT, im_DATA = im.size[0], im.size[1], im.convert('RGBA').tobytes("raw", "RGBA", 0, -1)
    self.depth_texture = self.ctx.texture((im.size[0], im.size[1]), components=4, data=im_DATA)
    self.depth_texture_uniform = self.prog.get('depth_texture', FakeUniform())
    self.depth_set = True
    
    
  def clear(self):
    self.fbo.clear()
    if self.bck_fbo:
      self.bck_fbo.clear()

  def extract_img(self, path):
    image = Image.frombytes('RGBA', self.size, self.fbo.read(components=4))
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(path, format='png')

  def render(self,**kwargs):
    self.fbo.use()

    for (name, texture) in self.textures.items():
      #print(name, texture, self.texture_uniforms[name].value)
      textureid = self.texture_uniforms[name].value
      self.textures[name].use(textureid)
      self.texture_uniforms[name].value = textureid

    if self.enable_backbuffer == True:
      self.bck_texture.use(1)
      self.backbuffer_uniform.value = 1
    if self.init_texture:
      self.init_texture.use(2)
      self.init_texture_uniform.value = 2
    if hasattr(self, 'depth_texture'):
      self.depth_texture.use(3)
      self.depth_texture_uniform.value = 3
    self.vao.render(moderngl.TRIANGLE_STRIP)
    
    if self.enable_backbuffer == True:
      self.bck_fbo.use()
      self.texture.use(0)
      self.copy_target_uniform.value = 0
      self.bck_vao.render(moderngl.TRIANGLE_STRIP)
    
    if 'time' in kwargs:
      self.time.value = kwargs['time']
    else:
      self.time.value = self.time.value + 0.01

    self.init.value = 0.0




