from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput

from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.base import runTouchApp

class MainScreen(FloatLayout):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)   
        
        self.parameters = Label(text='Parameters', size_hint=(0.2, 0.15), pos=(50, 500), font_size='30sp')
        self.add_widget(self.parameters)
        
        self.nuclear_species = Button(text='Nuclear species', size_hint=(0.15, 0.075), pos=(50, 450))
        self.add_widget(self.nuclear_species)

        self.nucleus_dd = DropDown()
        self.nuclear_species.bind(on_release=self.nucleus_dd.open)
        self.Cl_btn = Button(text='Cl', size_hint_y=None, height=25)
        self.Cl_btn.bind(on_release=lambda btn: self.nucleus_dd.select(btn.text))
        self.nucleus_dd.add_widget(self.Cl_btn)
        self.Na_btn = Button(text='Na', size_hint_y=None, height=25)
        self.Na_btn.bind(on_release=lambda btn: self.nucleus_dd.select(btn.text))
        self.nucleus_dd.add_widget(self.Na_btn)
        self.nucleus_dd.bind(on_select=lambda instance, x: setattr(self.nuclear_species, 'text', x))
        

class PulseBit(App):
    def build(self):
        return MainScreen(size=(500, 500))
    
PulseBit().run()