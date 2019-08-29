# -*- coding: utf-8 -*
"""
Kivy implementation of 1D convection.

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
        self.convection_1d = np.full(array_length, .5, dtype=np.float32)
        self.convection_1d[array_length // 4 : 3 * array_length // 4] = .75

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'r':  #Reset
            self.convection_1d = np.full(array_length, .5, dtype=np.float32)
            self.convection_1d[array_length // 4 : 3 * array_length // 4] = .75
        return True

    def update(self, dt):
        self.convection_1d -= self.convection_1d *\
                              nd.convolve1d(self.convection_1d, [0, 1, -1],
                                            mode='wrap')

        self.line.points = [coor
                            for x, y in enumerate(self.convection_1d)
                            for coor in [x * self.width / array_length,
                                         self.height * y]]
        return True

    def poke(self, poke_x, poke_y):
        scaled_x = int(poke_x * array_length / self.width)
        scaled_y = poke_y / self.height
        self.convection_1d[scaled_x - 2:scaled_x + 3] = scaled_y
        return True

    def on_touch_down(self, touch):
        self.poke(touch.x, touch.y)
        return True

    def on_touch_move(self, touch):
        self.poke(touch.x, touch.y)
        return True


class Convection_1D(App):
    def build(self):
        display = Display()
        Clock.schedule_interval(display.update, 1.0/120.0)
        return display


if __name__ == '__main__':
    Convection_1D().run()
