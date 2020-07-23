import numpy as np
from fractions import Fraction

from kivy.config import Config
Config.set('graphics', 'resizable', False)

from functools import partial

from kivy.app import App

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout

from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.app import runTouchApp

from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider

# Function which specifies the action of various controls (stub)
def on_enter():
        pass
    
# Function which automatically replaces the content of a TextInput with the new string 'text'
def clear_and_write_text(text_object, text, *args):
    text_object.select_all()
    text_object.delete_selection()
    text_object.insert_text(text)


# Class of the page of the software which lists the parameters of the system
class System_Parameters(FloatLayout):
    
    manual_dm = Widget()
    
    # Specifies the action of the checkbox 'Canonical', i.e. to toggle the TextInput widgets associated
    # with the temperature and the density matrix to be inserted manually respectively
    def on_canonical_active(self, *args):
        self.temperature.disabled = not self.temperature.disabled
        
        if self.spin_qn.text != '':
            for i in range(self.d):
                for j in range(self.d):
                    self.dm_elements[i, j].disabled = not self.dm_elements[i, j].disabled
    
    # Specifies the action carried out after the validation of the spin quantum number
    def set_quantum_number(self, *args):
        self.remove_widget(self.manual_dm)
        
        self.d = int(Fraction(self.spin_qn.text)*2+1)
        
        # Sets the grid representing the initial density matrix to be filled manually
        self.manual_dm = GridLayout(cols=self.d, size=(40*self.d, 40*self.d), size_hint=(None, None), pos=(50, 440-self.d*40))
        self.dm_elements = np.empty((self.d, self.d), dtype=TextInput)
        for j in range(self.d):
            for i in range(self.d):
                self.dm_elements[i, j] = TextInput(multiline=False, disabled=False)
                self.manual_dm.add_widget(self.dm_elements[i, j])
        self.add_widget(self.manual_dm)
    
    # Controls of the nuclear spin parameters
    def nuclear_parameters(self, x_shift=0, y_shift=0):        
        # Nuclear species dropdown list
        self.nuclear_species = Button(text='Nuclear species', size_hint=(0.15, 0.055), pos=(x_shift+50, y_shift+450))
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
        
        # Spin quantum number
        self.spin_qn_label = Label(text='Spin quantum number', size=(10, 5), pos=(x_shift-130, y_shift-25), font_size='15sp')
        self.add_widget(self.spin_qn_label)
        
        self.spin_qn = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+355, y_shift+460))
        self.spin_qn.bind(on_text_validate=self.set_quantum_number)
        self.add_widget(self.spin_qn)
        self.Cl_btn.bind(on_release=partial(clear_and_write_text, self.spin_qn, '3/2'))
        self.Na_btn.bind(on_release=partial(clear_and_write_text, self.spin_qn, '3/2'))
        
        # Gyromagnetic Ratio
        self.gyro_label = Label(text='Gyromagnetic ratio', size=(10, 5), pos=(x_shift+100, y_shift-25), font_size='15sp')
        self.add_widget(self.gyro_label)
        
        self.gyro = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+575, y_shift+460))
        self.gyro.bind(on_text_validate=on_enter)
        self.add_widget(self.gyro)
        self.Cl_btn.bind(on_release=partial(clear_and_write_text, self.gyro, '4.00'))
        self.Na_btn.bind(on_release=partial(clear_and_write_text, self.gyro, '11.26'))
        
        self.gyro_unit_label = Label(text='x 10\N{SUPERSCRIPT TWO} 1/Gs', size=(10, 5), pos=(x_shift+275, y_shift-25), font_size='15sp')
        self.add_widget(self.gyro_unit_label)

    # Controls of the magnetic field parameters
    def magnetic_parameters(self, x_shift, y_shift):
        
        self.mag_field = Label(text='Magnetic field', size=(10, 5), pos=(x_shift-285, y_shift+100), font_size='20sp')
        self.add_widget(self.mag_field)
        
        # Field magnitude
        self.field_mag_label = Label(text='Field magnitude', size=(10, 5), pos=(x_shift-296, y_shift+50), font_size='15sp')
        self.add_widget(self.field_mag_label)
        
        self.field_mag = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+170, y_shift+535))
        self.field_mag.bind(on_text_validate=on_enter)
        self.add_widget(self.field_mag)
        
        self.field_mag_unit = Label(text='G', size=(10, 5), pos=(x_shift-155, y_shift+50), font_size='15sp')
        self.add_widget(self.field_mag_unit)
        
        # Polar angle
        self.theta_z_label = Label(text='\N{GREEK SMALL LETTER THETA}z', size=(10, 5), pos=(x_shift-110, y_shift+50), font_size='15sp')
        self.add_widget(self.theta_z_label)
        
        self.theta_z = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+310, y_shift+535))
        self.theta_z.bind(on_text_validate=on_enter)
        self.add_widget(self.theta_z)
        
        self.theta_z_unit = Label(text='rad', size=(10, 5), pos=(x_shift-10, y_shift+50), font_size='15sp')
        self.add_widget(self.theta_z_unit)
        
        # Azimuthal angle
        self.phi_z_label = Label(text='\N{GREEK SMALL LETTER PHI}z', size=(10, 50), pos=(x_shift+45, y_shift+50), font_size='15sp')
        self.add_widget(self.phi_z_label)
        
        self.phi_z = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+470, y_shift+535))
        self.phi_z.bind(on_text_validate=on_enter)
        self.add_widget(self.phi_z)
        
        self.phi_z_unit = Label(text='rad', size=(10, 5), pos=(x_shift+150, y_shift+50), font_size='15sp')
        self.add_widget(self.phi_z_unit)
        
    # Controls of the quadrupole interaction parameters
    def quadrupole_parameters(self, x_shift=0, y_shift=0):
        self.quad_int = Label(text='Quadrupole interaction', size=(10, 5), pos=(x_shift-250, y_shift+90), font_size='20sp')
        self.add_widget(self.quad_int)
        
        # Coupling constant
        self.coupling_label = Label(text='Coupling constant e\N{SUPERSCRIPT TWO}qQ', size=(10, 5), pos=(x_shift-271, y_shift+40), font_size='15sp')
        self.add_widget(self.coupling_label)
        
        self.coupling = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+220, y_shift+525))
        self.coupling.bind(on_text_validate=on_enter)
        self.add_widget(self.coupling)
        
        self.coupling = Label(text='MHz', size=(10, 5), pos=(x_shift-95, y_shift+40), font_size='15sp')
        self.add_widget(self.coupling)
        
        # Asymmetry parameter
        self.asymmetry_label = Label(text='Asymmetry parameter', size=(10, 5), pos=(x_shift+25, y_shift+40), font_size='15sp')
        self.add_widget(self.asymmetry_label)
        
        self.asymmetry = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+515, y_shift+525))
        self.asymmetry.bind(on_text_validate=on_enter)
        self.add_widget(self.asymmetry)
        
        # Euler angles
        self.alpha_q_label = Label(text='\N{GREEK SMALL LETTER ALPHA}Q', size=(10, 5), pos=(x_shift-341, y_shift-10), font_size='15sp')
        self.add_widget(self.alpha_q_label)
        
        self.alpha_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+80, y_shift+475))
        self.alpha_q.bind(on_text_validate=on_enter)
        self.add_widget(self.alpha_q)
        
        self.alpha_q_unit = Label(text='rad', size=(10, 5), pos=(x_shift-240, y_shift-10), font_size='15sp')
        self.add_widget(self.alpha_q_unit)
        
        self.beta_q_label = Label(text='\N{GREEK SMALL LETTER BETA}Q', size=(10, 5), pos=(x_shift-180, y_shift-10), font_size='15sp')
        self.add_widget(self.beta_q_label)
        
        self.beta_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+241, y_shift+475))
        self.beta_q.bind(on_text_validate=on_enter)
        self.add_widget(self.beta_q)
        
        self.beta_q_unit = Label(text='rad', size=(10, 5), pos=(x_shift-79, y_shift-10), font_size='15sp')
        self.add_widget(self.beta_q_unit)
        
        self.gamma_q_label = Label(text='\N{GREEK SMALL LETTER GAMMA}Q', size=(10, 5), pos=(x_shift-19, y_shift-10), font_size='15sp')
        self.add_widget(self.gamma_q_label)
        
        self.gamma_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+402, y_shift+475))
        self.gamma_q.bind(on_text_validate=on_enter)
        self.add_widget(self.gamma_q)
       
        self.gamma_q_unit = Label(text='rad', size=(10, 5), pos=(x_shift+83, y_shift-10), font_size='15sp')
        self.add_widget(self.gamma_q_unit)
        
    # Controls on the initial density matrix
    def initial_dm_parameters(self, x_shift, y_shift):
    
        self.initial_dm = Label(text='Initial density matrix', size=(10, 5), pos=(x_shift-260, y_shift+30), font_size='20sp')
        self.add_widget(self.initial_dm)
        
        self.dm_par = GridLayout(cols=5, size=(500, 35), size_hint=(None, None), pos=(x_shift+50, y_shift+460))
        
        # Checkbox to set the initial density matrix as the canonical one
        self.canonical_checkbox = CheckBox(size_hint_x=None, width=20)
        self.dm_par.add_widget(self.canonical_checkbox)
        
        self.canonical_label = Label(text='Canonical', font_size='15sp', size_hint_x=None, width=100)
        self.dm_par.add_widget(self.canonical_label)
        
        # Temperature of the system (required if the canonical checkbox is active)
        self.temperature_label = Label(text='Temperature (K)', font_size='15sp', size_hint_x=None, width=150)
        self.dm_par.add_widget(self.temperature_label)
        
        self.temperature = TextInput(multiline=False, disabled=True, size_hint_x=None, width=70, size_hint_y=None, height=35)
        self.dm_par.add_widget(self.temperature)
        
        self.temperature_unit = Label(text='K', font_size='15sp', size_hint_x=None, width=30)
        self.dm_par.add_widget(self.temperature_unit)
        
        self.canonical_checkbox.bind(active=self.on_canonical_active)
        
        self.add_widget(self.dm_par)
        
    def __init__(self, **kwargs):
        super(System_Parameters, self).__init__(**kwargs)
        
        # Label 'System parameters'
        self.parameters = Label(text='System parameters', size=(10, 5), pos=(0, 450), font_size='30sp')
        self.add_widget(self.parameters)
        
        self.nuclear_parameters(0, 400)
        
        self.magnetic_parameters(0, 200)
        
        self.quadrupole_parameters(0, 100)
        
        self.initial_dm_parameters(0, 0)
        

# Class of the object on top of the individual panels
class Panels(TabbedPanel):
    def __init__(self, **kwargs):
        super(Panels, self).__init__(**kwargs)
                
        self.tab_sys_par = TabbedPanelItem(text='System')
        self.scroll_window =  ScrollView(size_hint=(1, None), size=(Window.width, 500))
        self.sys_par = System_Parameters(size_hint=(1, None), size=(Window.width, 1000))
        
        self.scroll_window.add_widget(self.sys_par)
        self.tab_sys_par.add_widget(self.scroll_window)
        self.add_widget(self.tab_sys_par)
        
        self.tab_pulse_par = TabbedPanelItem(text='Pulse')
        self.add_widget(self.tab_pulse_par)
        
        self.tab_spectrum = TabbedPanelItem(text='Spectrum')
        self.add_widget(self.tab_spectrum)
        
        self.tab_dm = TabbedPanelItem(text='State')
        self.add_widget(self.tab_dm)
        
        
# Class of the application
class PulseBit(App):
    def build(self):
        return Panels(size=(500, 500), do_default_tab=False, tab_pos='top_mid')
    
PulseBit().run()