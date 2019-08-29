# -*- coding: utf-8 -*
"""
Kivy implementation of 1D diffusion.

click to displace line
'r' to reset
"""
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics import Line
from kivy.core.window import Window
import numpy as np
import scipy.ndimage as nd

array_length = 512

class Display(Widget):
    def __init__(self, **kwargs):
        super(Display, self).__init__(**kwargs)
        with self.canvas:
            self.line = Line()
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.kernels_1d = [np.array([1, 0, 0]),
                           np.array([.5, 0, .5]),
                           np.array([1/3, 1/3, 1/3]),
                           np.array([.25, .5, .25]),
                           np.array([.1, .2, .4, .2, .1])]
        self.kernel = 1
        self.damping = 1.
        self.diffusion_1d = np.zeros(array_length, dtype=np.float32)
        self.diffusion_1d[array_length // 4 : 3 * array_length // 4] = .5

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'r':  #Reset
            self.diffusion_1d = np.zeros(array_length, dtype=np.float32)
            self.diffusion_1d[array_length // 4 : 3 * array_length // 4] = .5
            self.damping = 1.
        if keycode[1] == 'left': #Change kernel
            self.kernel = (self.kernel - 1) % len(self.kernels_1d)
        if keycode[1] == 'right':
            self.kernel = (self.kernel + 1) % len(self.kernels_1d)
        if keycode[1] == 'up':  #Increase damping
            self.damping -= .001
        if keycode[1] == 'down': #Decrease damping
            self.damping += .001
            if self.damping > 1: self.damping = 1.
        return True

    def update(self, dt):
        self.diffusion_1d = self.damping *\
                            nd.convolve1d(self.diffusion_1d,
                                          self.kernels_1d[self.kernel],
                                          mode='wrap')
        with self.canvas:
            self.canvas.remove(self.line)
            self.line = Line(points=[coor
                                     for x, y in enumerate(self.diffusion_1d)
                                     for coor in [x * self.width / array_length,
                                                  self.height // 2 * (y + 1)]])
        return True

    def poke(self, poke_x, poke_y):
        scaled_x = int(poke_x * array_length / self.width)
        scaled_y = poke_y * 2 / self.height - 1
        self.diffusion_1d[scaled_x - 2:scaled_x + 3] = scaled_y
        return True

    def on_touch_down(self, touch):
        self.poke(touch.x, touch.y)
        return True

    def on_touch_move(self, touch):
        self.poke(touch.x, touch.y)
        return True


class Diffusion_1D(App):
    def build(self):
        display = Display()
        Clock.schedule_interval(display.update, 1.0/60.0)
        return display


if __name__ == '__main__':
    Diffusion_1D().run()
