from functools import partial

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout

from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.base import runTouchApp


def on_enter(instance, value):
    print('User pressed enter in', instance)


def clear_and_write_text(text_object, text, *args):
    text_object.select_all()
    text_object.delete_selection()
    text_object.insert_text(text)


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
        
        self.spin_qn_label = Label(text='Spin quantum number', size_hint=(0.1, 0.05), pos=(225, 457.5), font_size='15sp')
        self.add_widget(self.spin_qn_label)
        
        self.spin_qn = TextInput(multiline=False, size_hint=(0.05, 0.05), pos=(350, 457.5))
        self.spin_qn.bind(on_text_validate=on_enter)
        self.add_widget(self.spin_qn)
        self.Cl_btn.bind(on_release=partial(clear_and_write_text, self.spin_qn, '3/2'))
        self.Na_btn.bind(on_release=partial(clear_and_write_text, self.spin_qn, '3/2'))
        
        self.gyro_label = Label(text='Gyromagnetic ratio', size_hint=(0.1, 0.05), pos=(430, 457.5), font_size='15sp')
        self.add_widget(self.gyro_label)
        
        self.gyro = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(547.5, 457.5))
        self.gyro.bind(on_text_validate=on_enter)
        self.add_widget(self.gyro)
        self.Cl_btn.bind(on_release=partial(clear_and_write_text, self.gyro, '4.00'))
        self.Na_btn.bind(on_release=partial(clear_and_write_text, self.gyro, '11.26'))
        
        self.gyro_unit_label = Label(text='x 10\N{SUPERSCRIPT TWO} 1/Gs', size_hint=(0.1, 0.05), pos=(600, 457.5), font_size='15sp')
        self.add_widget(self.gyro_unit_label)
        

class PulseBit(App):
    def build(self):
        return MainScreen(size=(500, 500))
    
PulseBit().run()