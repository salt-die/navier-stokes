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
        self.burgers_2d = np.zeros(texture_dim, dtype=np.float32).T
        self.burgers_2d[texture_dim[0] // 4 : 3 * texture_dim[0] // 4,
                          texture_dim[1] // 4 : 3 * texture_dim[1] // 5] = 1

    def _update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'r':  #Reset
            self.burgers_2d = np.zeros(texture_dim, dtype=np.float32).T
            self.burgers_2d[texture_dim[0] // 4 : 3 * texture_dim[0] // 4,
                               texture_dim[1] // 4 : 3 * texture_dim[1] // 5] = 1
        return True

    def update(self, dt):
        con_kernel = np.array([[   0, .25,    0],
                               [ .25,  -1,  .25],
                               [   0, .25,    0]])
        dif_kernel = np.array([[.025,  .1, .025],
                               [  .1,  .5,   .1],
                               [.025,  .1, .025]])
        con_constant =  .74 #convection constant
        self.burgers_2d = con_constant * self.burgers_2d *\
                          nd.convolve(self.burgers_2d, con_kernel,
                                      mode='wrap') +\
                          nd.convolve(self.burgers_2d, dif_kernel,
                                      mode='wrap')

        self.texture.blit_buffer(np.dstack([np.zeros(texture_dim,
                                               dtype=np.float32)] * 2 +\
                                               [self.burgers_2d]).tobytes(),
                                 colorfmt='rgb', bufferfmt='float')
        self.canvas.ask_update()
        return True

    def poke(self, poke_x, poke_y):
        scaled_x = int(poke_x * texture_dim[0] / self.width)
        scaled_y = int(poke_y * texture_dim[1] / self.height)
        self.burgers_2d[scaled_y - 5:scaled_y + 6,
                           scaled_x - 5:scaled_x + 6] = 1
        return True

    def on_touch_down(self, touch):
        self.poke(touch.x, touch.y)
        return True

    def on_touch_move(self, touch):
        self.poke(touch.x, touch.y)
        return True


class Burgers_2D(App):
    def build(self):
        display = Display()
        Clock.schedule_interval(display.update, 1.0/120.0)
        return display


if __name__ == '__main__':
    Burgers_2D().run()
