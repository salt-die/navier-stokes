# -*- coding: utf-8 -*
"""
Kivy implementation of 1D burgers.

click to displace line
'r' to reset
"""
import numpy as np
import scipy.ndimage as nd
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.core.window import Window

texture_dim = [256, 256]
#boundary condition - 'wrap', 'reflect', 'constant', 'nearest', 'mirror'
bc = "wrap"
viscosity = .08  #Is it odd that negative viscosity still works?
rho = 1.99  #Density
damping = .994
external_flow = .38  #flow in the horizontal direction

#drop just makes pokes look a little better
drop = np.array([[0., 0., 1., 1., 1., 1., 1., 0., 0.],\
                 [0., 1., 1., 1., 1., 1., 1., 1., 0.],\
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                 [0., 1., 1., 1., 1., 1., 1., 1., 0.],\
                 [0., 0., 1., 1., 1., 1., 1., 0., 0.],])

red = np.zeros(texture_dim, dtype=np.float32)
green = np.full(texture_dim, .6549, dtype=np.float32)

con_kernel = np.array([[   0, .25,    0],
                       [ .25,  -1,  .25],
                       [   0, .25,    0]])
dif_kernel = np.array([[.025,  .1, .025],
                       [  .1,  .5,   .1],
                       [.025,  .1, .025]])
poi_kernel = np.array([[   0, .25,    0],
                       [ .25,   0,  .25],
                       [   0, .25,    0]])
flow_kernal = np.array([[0, 0, 0],
                        [-external_flow, 1, external_flow],
                        [0, 0, 0]])

class Display(Widget):
    def __init__(self, **kwargs):
        super(Display, self).__init__(**kwargs)
        self.texture = Texture.create(size=texture_dim)
        with self.canvas:
            self.rect = Rectangle(texture=self.texture, pos=self.pos,
                                  size=(self.width, self.height))
        self.bind(size=self._update_rect, pos=self._update_rect)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.reset()

    def reset(self):
        self.momentum = np.zeros(texture_dim, dtype=np.float32).T
        self.momentum[3 * texture_dim[0] // 8 : 5 * texture_dim[0] // 8,
                      3 * texture_dim[1] // 8 : 5 * texture_dim[1] // 8] = .04
        self.pressure = np.zeros(texture_dim, dtype=np.float32).T
        self.pressure[3 * texture_dim[0] // 8 : 5 * texture_dim[0] // 8,
                      3 * texture_dim[1] // 8 : 5 * texture_dim[1] // 8] = 1
        self.walls = np.zeros(texture_dim, dtype=np.float32).T

    def _update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'r':
            self.reset()
        return True

    def update(self, dt):
        self.momentum = (nd.convolve(self.momentum, dif_kernel, mode=bc) -\
                        viscosity * self.momentum *\
                        nd.convolve(self.momentum, con_kernel, mode=bc) +\
                        nd.convolve(self.pressure, con_kernel, mode=bc) *\
                        rho / 2) * damping

        if external_flow:
            self.momentum = nd.convolve(self.momentum, flow_kernal, mode=bc)

        #dif for difference, not diffusion -- dif is the change in momentum
        dif = nd.convolve(self.momentum, poi_kernel, mode=bc)

        self.pressure = (nd.convolve(self.pressure, poi_kernel, mode=bc) +\
                        rho / 4 * (dif - dif**2)) * damping

        #Add some noise for a bit a of realism
        self.pressure += np.random.normal(scale=.003, size=texture_dim).T
        self.momentum += np.random.normal(scale=.003, size=texture_dim).T

        #Keep the values from running away
        np.clip(self.pressure, -1, 1, out=self.pressure)
        np.clip(self.momentum, -1, 1, out=self.momentum)

        #Wall boundary conditions
        self.momentum = np.where(self.walls !=1, self.momentum, -.45)
        self.pressure = np.where(self.walls !=1, self.pressure, 0.1)

        #Blit
        RGB = np.dstack([red, green * self.pressure, (self.pressure + 1) * .5])
        RGB[self.walls == 1] = np.array([.717, .176, .07])
        self.texture.blit_buffer(RGB.tobytes(), colorfmt='rgb',
                                 bufferfmt='float')
        self.canvas.ask_update()
        return True

    def poke(self, touch):
        scaled_x = int(touch.x * texture_dim[0] / self.width)
        scaled_y = int(touch.y * texture_dim[1] / self.height)
        try:
            if touch.button == "left":
                self.pressure[scaled_y - 4:scaled_y + 5,
                              scaled_x - 4:scaled_x + 5][drop == 1] = .5
                self.momentum[scaled_y - 4:scaled_y + 5,
                              scaled_x - 4:scaled_x + 5][drop == 1] = .1
            if touch.button == "right":
                self.walls[scaled_y - 4:scaled_y + 5,
                           scaled_x - 4:scaled_x + 5][drop == 1] = 1
        except:
            #Too close to border.
            pass
        return True

    def on_touch_down(self, touch):
        self.poke(touch)
        return True

    def on_touch_move(self, touch):
        self.poke(touch)
        return True


class Navier_Stokes(App):
    def build(self):
        display = Display()
        Clock.schedule_interval(display.update, 1.0/120.0)
        return display


if __name__ == '__main__':
    Navier_Stokes().run()
