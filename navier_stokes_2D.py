# -*- coding: utf-8 -*
"""
Kivy implementation of 1D burgers.

click to displace line
'r' to reset
"""
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.core.window import Window
import numpy as np
import scipy.ndimage as nd

texture_dim = [256, 256]

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
        self.momentum = np.zeros(texture_dim, dtype=np.float32).T
        self.pressure = np.zeros(texture_dim, dtype=np.float32).T
        self.pressure[texture_dim[0] // 4 : 3 * texture_dim[0] // 4,
                      texture_dim[1] // 4 : 3 * texture_dim[1] // 5] = 1
        self.two_thirds_stack = [np.zeros(texture_dim, dtype=np.float32)] * 2

    def _update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'r':  #Reset
            self.momentum = np.zeros(texture_dim, dtype=np.float32).T
            self.pressure = np.zeros(texture_dim, dtype=np.float32).T
            self.pressure[texture_dim[0] // 4 : 3 * texture_dim[0] // 4,
                          texture_dim[1] // 4 : 3 * texture_dim[1] // 5] = 1
        return True

    def update(self, dt):
        con_kernel = np.array([[   0, .25,    0],
                               [ .25,  -1,  .25],
                               [   0, .25,    0]])
        dif_kernel = np.array([[.025,  .1, .025],
                               [  .1,  .5,   .1],
                               [.025,  .1, .025]])
        poi_kernel = np.array([[   0, .25,    0],
                               [ .25,   0,  .25],
                               [   0, .25,    0]])
        #boundary condition - 'wrap', 'reflect', 'constant', 'nearest', 'mirror'
        bc = "wrap"
        viscosity = .001  #Is it odd that negative viscosity still works?
        rho = 2.1  #Density -- Between -2 and 2 is reasonable
        damping = .994
        external_flow = .4 #flow in the horizontal direction
        flow_kernal = np.array([[0, 0, 0],
                                [-external_flow, 1, external_flow],
                                [0, 0, 0]])

        self.momentum = (viscosity * self.momentum *\
                        nd.convolve(self.momentum, con_kernel, mode=bc) +\
                        nd.convolve(self.momentum, dif_kernel, mode=bc) -\
                        nd.convolve(self.pressure, con_kernel, mode=bc) *\
                        (1 / 2 * rho)) * damping
        if external_flow:
            self.momentum = nd.convolve(self.momentum, flow_kernal, mode=bc)
        #dif for difference, not diffusion -- dif is the change in momentum
        dif = nd.convolve(self.momentum, poi_kernel, mode=bc)
        self.pressure = (nd.convolve(self.pressure, poi_kernel, mode=bc) -\
                        rho / 4 * (dif - dif**2)) * damping

        #Add some noise for a bit a of realism
        self.pressure += np.random.normal(scale=.005, size=texture_dim).T
        self.momentum += np.random.normal(scale=.005, size=texture_dim).T

        #Keep the values from running away
        np.clip(self.pressure, -2, 2, out=self.pressure)
        np.clip(self.momentum, -2, 2, out=self.momentum)

        self.texture.blit_buffer(np.dstack(self.two_thirds_stack +\
                                           [(self.pressure + 1) / 2]).tobytes(),
                                 colorfmt='rgb', bufferfmt='float')
        self.canvas.ask_update()
        return True

    def poke(self, poke_x, poke_y):
        scaled_x = int(poke_x * texture_dim[0] / self.width)
        scaled_y = int(poke_y * texture_dim[1] / self.height)
        self.pressure[scaled_y - 5:scaled_y + 6,
                      scaled_x - 5:scaled_x + 6] = 1
        self.momentum[scaled_y - 5:scaled_y + 6,
                      scaled_x - 5:scaled_x + 6] = 0
        return True

    def on_touch_down(self, touch):
        self.poke(touch.x, touch.y)
        return True

    def on_touch_move(self, touch):
        self.poke(touch.x, touch.y)
        return True


class Navier_Stokes(App):
    def build(self):
        display = Display()
        Clock.schedule_interval(display.update, 1.0/120.0)
        return display


if __name__ == '__main__':
    Navier_Stokes().run()