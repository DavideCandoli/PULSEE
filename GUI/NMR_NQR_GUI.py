from kivy.config import Config
Config.set('graphics', 'resizable', False)

from functools import partial

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout

from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

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


# Class of the page of the software which lists the parameters of the system
class System_Parameters(FloatLayout):
    def __init__(self, **kwargs):
        super(System_Parameters, self).__init__(**kwargs)
        
        # Label 'System parameters'
        self.parameters = Label(text='System parameters', size_hint=(0.2, 0.15), pos=(100, 485), font_size='30sp')
        self.add_widget(self.parameters)
        
        # Controls of the nuclear spin parameters
        # Nuclear species dropdown list
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
        
        # Spin quantum number
        self.spin_qn_label = Label(text='Spin quantum number', size_hint=(0.1, 0.05), pos=(225, 457.5), font_size='15sp')
        self.add_widget(self.spin_qn_label)
        
        self.spin_qn = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(350, 457.5))
        self.spin_qn.bind(on_text_validate=on_enter)
        self.add_widget(self.spin_qn)
        self.Cl_btn.bind(on_release=partial(clear_and_write_text, self.spin_qn, '3/2'))
        self.Na_btn.bind(on_release=partial(clear_and_write_text, self.spin_qn, '3/2'))
        
        # Gyromagnetic Ratio
        self.gyro_label = Label(text='Gyromagnetic ratio', size_hint=(0.1, 0.05), pos=(450, 457.5), font_size='15sp')
        self.add_widget(self.gyro_label)
        
        self.gyro = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(565, 457.5))
        self.gyro.bind(on_text_validate=on_enter)
        self.add_widget(self.gyro)
        self.Cl_btn.bind(on_release=partial(clear_and_write_text, self.gyro, '4.00'))
        self.Na_btn.bind(on_release=partial(clear_and_write_text, self.gyro, '11.26'))
        
        self.gyro_unit_label = Label(text='x 10\N{SUPERSCRIPT TWO} 1/Gs', size_hint=(0.1, 0.05), pos=(620, 457.5), font_size='15sp')
        self.add_widget(self.gyro_unit_label)
        
        # Label 'Magnetic field'
        self.mag_field = Label(text='Magnetic field', size_hint=(0.165, 0.15), pos=(50, 360), font_size='20sp')
        self.add_widget(self.mag_field)
        
        # Controls of the magnetic field parameters
        self.field_mag_label = Label(text='Field magnitude', size_hint=(0.135, 0.1), pos=(50, 330), font_size='15sp')
        self.add_widget(self.field_mag_label)
        
        self.field_mag = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(170, 345))
        self.field_mag.bind(on_text_validate=on_enter)
        self.add_widget(self.field_mag)
        
        self.field_mag_unit = Label(text='G', size_hint=(0.135, 0.1), pos=(180, 330), font_size='15sp')
        self.add_widget(self.field_mag_unit)
        
        self.theta_z_label = Label(text='\N{GREEK SMALL LETTER THETA}z', size_hint=(0.135, 0.1), pos=(220, 330), font_size='15sp')
        self.add_widget(self.theta_z_label)
        
        self.theta_z = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(295, 345))
        self.theta_z.bind(on_text_validate=on_enter)
        self.add_widget(self.theta_z)
        
        self.theta_z_unit = Label(text='rad', size_hint=(0.135, 0.1), pos=(310, 330), font_size='15sp')
        self.add_widget(self.theta_z_unit)
        
        self.phi_z_label = Label(text='\N{GREEK SMALL LETTER PHI}z', size_hint=(0.135, 0.1), pos=(360, 330), font_size='15sp')
        self.add_widget(self.phi_z_label)
        
        self.phi_z = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(435, 345))
        self.phi_z.bind(on_text_validate=on_enter)
        self.add_widget(self.phi_z)
        
        self.phi_z_unit = Label(text='rad', size_hint=(0.135, 0.1), pos=(450, 330), font_size='15sp')
        self.add_widget(self.phi_z_unit)
        
        # Label 'Quadrupole interaction'
        self.quad_int = Label(text='Quadrupole interaction', size_hint=(0.165, 0.15), pos=(87.5, 260), font_size='20sp')
        self.add_widget(self.quad_int)
        
        # Controls of the quadrupole interaction parameters
        self.coupling_label = Label(text='Coupling constant e\N{SUPERSCRIPT TWO}qQ', size_hint=(0.135, 0.1), pos=(76.5, 230), font_size='15sp')
        self.add_widget(self.coupling_label)
        
        self.coupling = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(220, 245))
        self.coupling.bind(on_text_validate=on_enter)
        self.add_widget(self.coupling)
        
        self.coupling = Label(text='MHz', size_hint=(0.135, 0.1), pos=(240, 230), font_size='15sp')
        self.add_widget(self.coupling)
        
        self.asymmetry_label = Label(text='Asymmetry parameter', size_hint=(0.135, 0.1), pos=(360, 230), font_size='15sp')
        self.add_widget(self.asymmetry_label)
        
        self.asymmetry = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(500, 245))
        self.asymmetry.bind(on_text_validate=on_enter)
        self.add_widget(self.asymmetry)
        
        self.alpha_q_label = Label(text='\N{GREEK SMALL LETTER ALPHA}Q', size_hint=(0.135, 0.1), pos=(5, 180), font_size='15sp')
        self.add_widget(self.alpha_q_label)
        
        self.alpha_q = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(80, 195))
        self.alpha_q.bind(on_text_validate=on_enter)
        self.add_widget(self.alpha_q)
        
        self.alpha_q_unit = Label(text='rad', size_hint=(0.135, 0.1), pos=(95, 180), font_size='15sp')
        self.add_widget(self.alpha_q_unit)
        
        self.beta_q_label = Label(text='\N{GREEK SMALL LETTER BETA}Q', size_hint=(0.135, 0.1), pos=(150, 180), font_size='15sp')
        self.add_widget(self.beta_q_label)
        
        self.beta_q = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(225, 195))
        self.beta_q.bind(on_text_validate=on_enter)
        self.add_widget(self.beta_q)
        
        self.beta_q_unit = Label(text='rad', size_hint=(0.135, 0.1), pos=(240, 180), font_size='15sp')
        self.add_widget(self.beta_q_unit)
        
        self.gamma_q_label = Label(text='\N{GREEK SMALL LETTER GAMMA}Q', size_hint=(0.135, 0.1), pos=(295, 180), font_size='15sp')
        self.add_widget(self.gamma_q_label)
        
        self.gamma_q = TextInput(multiline=False, size_hint=(0.065, 0.05), pos=(370, 195))
        self.gamma_q.bind(on_text_validate=on_enter)
        self.add_widget(self.gamma_q)
        
        self.gamma_q_unit = Label(text='rad', size_hint=(0.135, 0.1), pos=(385, 180), font_size='15sp')
        self.add_widget(self.gamma_q_unit)
        
        
# Class of the object on top of the individual panels
class Panels(TabbedPanel):
    def __init__(self, **kwargs):
        super(Panels, self).__init__(**kwargs)
        
        self.tab_sys_par = TabbedPanelItem(text='System')
        self.add_widget(self.tab_sys_par)
        self.tab_sys_par.content = System_Parameters()
        
        self.tab_pulse_par = TabbedPanelItem(text='Pulse')
        self.add_widget(self.tab_pulse_par)
        
        self.tab_spectrum = TabbedPanelItem(text='Spectrum')
        self.add_widget(self.tab_spectrum)
        
        self.tab_dm = TabbedPanelItem(text='State')
        self.add_widget(self.tab_dm)
        
        #tab_header = TabbedPanelHeader(text='Tab2')
        #tp.add_widget(th)

# Class of the application
class PulseBit(App):
    def build(self):
        return Panels(size=(500, 500), do_default_tab=False, tab_pos='top_mid')
    
PulseBit().run()