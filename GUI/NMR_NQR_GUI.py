import sys
import math
import numpy as np
import pandas as pd
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

from kivy.graphics import *


sys.path.insert(1, '/home/davidecandoli/Documenti/Università/Thesis/NQR-NMRSimulationSoftware')

from Simulation import Nuclear_System_Setup, \
                       Evolve, \
                       Transition_Spectrum, \
                       Plot_Real_Density_Matrix, \
                       Plot_Transition_Spectrum, \
                       FID_Signal, Plot_FID_Signal, \
                       Fourier_Transform_Signal, \
                       Plot_Fourier_Transform


# This class defines the object responsible of the management of the inputs and outputs of the
# simulation, mediating the interaction between the GUI and the computational core of the program.
class Simulation_Manager:
    spin_par = {'quantum number' : 0,
                'gyromagnetic ratio' : 0}
    
    zeem_par = {'field magnitude' : 0,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0,
                'asymmetry parameter' : 0,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    
    mode = pd.DataFrame([(0, 0, 0, 0, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    RRF_par = {'omega_RRF': 0,
               'theta_RRF': 0,
               'phi_RRF': 0}
    
    temperature = 300
    
    dm_0 = 0
    
sim_man = Simulation_Manager()


# Function which specifies the action of various controls (stub)
def on_enter():
        pass
    
def print_traceback(err, *args):
    raise err
    
# Function which automatically replaces the content of a TextInput with the new string 'text'
def clear_and_write_text(text_object, text, *args):
    text_object.select_all()
    text_object.delete_selection()
    text_object.insert_text(text)


# Class of the page of the software which lists the parameters of the system
class System_Parameters(FloatLayout):
    
    d = 1
    
    dm_elements = np.empty((d, d), dtype=Widget)
    dm_elements[0, 0] = TextInput()
    
    manual_dm = Widget()
    
    manual_dm_confirm = Widget()
    
    error_spin_qn = Label(text='')
    
    error_gyro = Label(text='')
    
    error_field_mag = Label(text='')
    
    error_theta_z = Label(text='')
    
    error_phi_z = Label(text='')
    
    error_e2qQ = Label(text='')
    
    error_eta = Label(text='')
    
    error_alpha_q = Label(text='')

    error_beta_q = Label(text='')

    error_gamma_q = Label(text='')
    
    error_canonical = Label(text='')
    
    error_T = Label(text='')
    
    error_manual_dm = Label(text='')
    
    error_build_system = Label(text='')
    
    tb_button = Button()
    
    # Specifies the action of the checkbox 'Canonical', i.e. to toggle the TextInput widgets associated
    # with the temperature, the density matrix to be inserted manually and the related button
    def on_canonical_active(self, *args):
        try:
            self.remove_widget(self.error_canonical)
            
            self.temperature.disabled = not self.temperature.disabled
        
            if self.spin_qn.text != '':
                for i in range(self.d):
                    for j in range(self.d):
                        self.dm_elements[i, j].disabled = not self.dm_elements[i, j].disabled
                self.manual_dm_confirm.disabled = not self.manual_dm_confirm.disabled
        except Exception as e:
            self.error_canonical=Label(text=e.args[0], pos=(-175, -50), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_canonical)
            
    
    # Specifies the action carried out after the validation of the spin quantum number
    def set_quantum_number(self, *args):
        try:
            self.remove_widget(self.error_spin_qn)
            
            self.remove_widget(self.manual_dm)
            
            sim_man.spin_par['quantum number'] = float(Fraction(self.spin_qn.text))
        
            self.d = int(Fraction(self.spin_qn.text)*2+1)
        
            if self.d <= 8: self.el_w = 40
            else: self.el_w = 30
        
        # Sets the grid representing the initial density matrix to be filled manually
            self.manual_dm = GridLayout(cols=self.d, size=(self.el_w*self.d, self.el_w*self.d), size_hint=(None, None), pos=(50, 405-self.d*self.el_w))
            self.dm_elements = np.empty((self.d, self.d), dtype=TextInput)
            for j in range(self.d):
                for i in range(self.d):
                    self.dm_elements[i, j] = TextInput(multiline=False, disabled=False)
                    self.manual_dm.add_widget(self.dm_elements[i, j])
                    self.dm_elements[i, j].disabled = not self.temperature.disabled
            self.add_widget(self.manual_dm)
            
            self.manual_dm_confirm = Button(text='Set density matrix', font_size='15sp', size_hint_y=None, height=30, size_hint_x=None, width=140, pos=(50, 420))
            self.manual_dm_confirm.disabled = not self.temperature.disabled
            self.manual_dm_confirm.bind(on_release=self.set_manual_dm)
            self.add_widget(self.manual_dm_confirm)
            
        # Prints any error raised after the validation of the spin quantum number below the TextInput
        except Exception as e:
            self.error_spin_qn=Label(text=e.args[0], pos=(-10, 337.5), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_spin_qn)
            
    # Specifies the action carried out after the validation of the spin's gyromagnetic ratio
    def set_gyro_ratio(self, *args):
        try:
            self.remove_widget(self.error_gyro)
            
            sim_man.spin_par['gyromagnetic_ratio'] = float(self.gyro.text)
            
        except Exception as e:
            self.error_gyro=Label(text=e.args[0], pos=(210, 337.5), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_gyro)
            
    # Specifies the action carried out after the validation of the magnetic field magnitude
    def set_field_magnitude(self, *args):
        try:
            self.remove_widget(self.error_field_mag)
            
            sim_man.zeem_par['field magnitude'] = float(self.field_mag.text)
            
        except Exception as e:
            self.error_field_mag=Label(text=e.args[0], pos=(-195, 220), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_field_mag)
            
    # Specifies the action carried out after the validation of the polar angle of the magnetic field
    def set_theta_z(self, *args):
        try:
            self.remove_widget(self.error_theta_z)
            
            sim_man.zeem_par['theta_z'] = (float(self.theta_z.text)*math.pi)/180
            
        except Exception as e:
            self.error_theta_z=Label(text=e.args[0], pos=(-60, 220), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_theta_z)
            
    # Specifies the action carried out after the validation of the azimuthal angle of the magnetic field
    def set_phi_z(self, *args):
        try:
            self.remove_widget(self.error_phi_z)
            
            sim_man.zeem_par['phi_z'] = (float(self.phi_z.text)*math.pi)/180
            
        except Exception as e:
            self.error_phi_z=Label(text=e.args[0], pos=(110, 220), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_phi_z)
            
    # Specifies the action carried out after the validation of the coupling constant of the quadrupole
    # interaction
    def set_coupling_constant(self, *args):
        try:
            self.remove_widget(self.error_e2qQ)
            
            sim_man.quad_par['coupling constant'] = float(self.coupling.text)

        except Exception as e:
            self.error_e2qQ=Label(text=e.args[0], pos=(-150, 117.5), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_e2qQ)
            
    # Specifies the action carried out after the validation of the asymmetry parameter of the EFG
    def set_asymmetry(self, *args):
        try:
            self.remove_widget(self.error_eta)
            
            sim_man.quad_par['asymmetry parameter'] = float(self.asymmetry.text)

        except Exception as e:
            self.error_eta=Label(text=e.args[0], pos=(150, 117.5), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_eta)
            
    # Specifies the action carried out after the validation of the Euler angle alpha_q
    def set_alpha_q(self, *args):
        try:
            self.remove_widget(self.error_alpha_q)
            
            sim_man.quad_par['alpha_q'] = (float(self.alpha_q.text)*math.pi)/180

        except Exception as e:
            self.error_alpha_q=Label(text=e.args[0], pos=(-250, 60), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_alpha_q)
            
    # Specifies the action carried out after the validation of the Euler angle beta_q
    def set_beta_q(self, *args):
        try:
            self.remove_widget(self.error_beta_q)
            
            sim_man.quad_par['beta_q'] = (float(self.beta_q.text)*math.pi)/180

        except Exception as e:
            self.error_beta_q=Label(text=e.args[0], pos=(-125, 60), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_beta_q)
            
    # Specifies the action carried out after the validation of the Euler angle gamma_q
    def set_gamma_q(self, *args):
        try:
            self.remove_widget(self.error_gamma_q)
            
            sim_man.quad_par['gamma_q'] = (float(self.gamma_q.text)*math.pi)/180

        except Exception as e:
            self.error_gamma_q=Label(text=e.args[0], pos=(40, 60), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_gamma_q)
            
    # Specifies the action carried out after the validation of the temperature
    def set_temperature(self, *args):
        try:
            self.remove_widget(self.error_T)
            
            sim_man.temperature = float(self.temperature.text)

        except Exception as e:
            self.error_T=Label(text=e.args[0], pos=(40, 60), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_T)
            
    # Specifies the action carried out after the confirmation of the density matrix specified manually
    def set_manual_dm(self, *args):
        try:
            self.remove_widget(self.error_manual_dm)
            
            sim_man.dm_0 = np.zeros((self.d, self.d), dtype=complex)
            
            for i in range(self.d):
                for j in range(self.d):
                    if self.dm_elements[i, j].text == "":
                        pass
                    else:
                        sim_man.dm_0[i, j] = complex(self.dm_elements[i, j].text)

        except Exception as e:
            self.error_manual_dm=Label(text=e.args[0], pos=(-70, -65), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_manual_dm)
    
    # Builds up the objects representing the nuclear system
    def build_system(self, *args):
        try:
            self.remove_widget(self.error_build_system)
            
            self.remove_widget(self.tb_button)
            
            sim_man.spin_par['quantum number'] = float(Fraction(self.spin_qn.text))
            
            sim_man.spin_par['gyromagnetic_ratio'] = float(self.gyro.text)
            
            sim_man.zeem_par['field magnitude'] = float(self.field_mag.text)
            
            sim_man.zeem_par['theta_z'] = (float(self.theta_z.text)*math.pi)/180
            
            sim_man.zeem_par['phi_z'] = (float(self.phi_z.text)*math.pi)/180
            
            sim_man.quad_par['coupling constant'] = float(self.coupling.text)
            
            sim_man.quad_par['asymmetry parameter'] = float(self.asymmetry.text)
            
            sim_man.quad_par['alpha_q'] = (float(self.alpha_q.text)*math.pi)/180
            
            sim_man.quad_par['beta_q'] = (float(self.beta_q.text)*math.pi)/180
            
            sim_man.quad_par['gamma_q'] = (float(self.gamma_q.text)*math.pi)/180
            
            if self.canonical_checkbox.active:
                sim_man.temperature = float(self.temperature.text)
            
            sim_man.dm_0 = np.zeros((self.d, self.d), dtype=complex)
            
            for i in range(self.d):
                for j in range(self.d):
                    if self.dm_elements[i, j].text == "":
                        pass
                    else:
                        sim_man.dm_0[i, j] = complex(self.dm_elements[i, j].text)
            
            if self.canonical_checkbox.active:
                Nuclear_System_Setup(sim_man.spin_par, \
                                     sim_man.zeem_par, \
                                     sim_man.quad_par, \
                                     initial_state='canonical', \
                                     temperature=sim_man.temperature)
            else:
                Nuclear_System_Setup(sim_man.spin_par, \
                                     sim_man.zeem_par, \
                                     sim_man.quad_par, \
                                     initial_state=sim_man.dm_0, \
                                     temperature=300)

        except Exception as e:
            self.error_build_system=Label(text=e.args[0], pos=(0, -490), size=(200, 200), bold=True, color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_build_system)
                        
            self.tb_button=Button(text='traceback', size_hint=(0.1, 0.03), pos=(450, 25))
            self.tb_button.bind(on_release=partial(print_traceback, e))
            self.add_widget(self.tb_button)
    
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
        self.gyro.bind(on_text_validate=self.set_gyro_ratio)
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
        self.field_mag.bind(on_text_validate=self.set_field_magnitude)
        self.add_widget(self.field_mag)
        
        self.field_mag_unit = Label(text='G', size=(10, 5), pos=(x_shift-155, y_shift+50), font_size='15sp')
        self.add_widget(self.field_mag_unit)
        
        # Polar angle
        self.theta_z_label = Label(text='\N{GREEK SMALL LETTER THETA}z', size=(10, 5), pos=(x_shift-110, y_shift+50), font_size='15sp')
        self.add_widget(self.theta_z_label)
        
        self.theta_z = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+310, y_shift+535))
        self.theta_z.bind(on_text_validate=self.set_theta_z)
        self.add_widget(self.theta_z)
        
        self.theta_z_unit = Label(text='°', size=(10, 5), pos=(x_shift-20, y_shift+50), font_size='15sp')
        self.add_widget(self.theta_z_unit)
        
        # Azimuthal angle
        self.phi_z_label = Label(text='\N{GREEK SMALL LETTER PHI}z', size=(10, 50), pos=(x_shift+45, y_shift+50), font_size='15sp')
        self.add_widget(self.phi_z_label)
        
        self.phi_z = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+470, y_shift+535))
        self.phi_z.bind(on_text_validate=self.set_phi_z)
        self.add_widget(self.phi_z)
        
        self.phi_z_unit = Label(text='°', size=(10, 5), pos=(x_shift+140, y_shift+50), font_size='15sp')
        self.add_widget(self.phi_z_unit)
        
    # Controls of the quadrupole interaction parameters
    def quadrupole_parameters(self, x_shift=0, y_shift=0):
        self.quad_int = Label(text='Quadrupole interaction', size=(10, 5), pos=(x_shift-250, y_shift+90), font_size='20sp')
        self.add_widget(self.quad_int)
        
        # Coupling constant
        self.coupling_label = Label(text='Coupling constant e\N{SUPERSCRIPT TWO}qQ', size=(10, 5), pos=(x_shift-271, y_shift+40), font_size='15sp')
        self.add_widget(self.coupling_label)
        
        self.coupling = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+220, y_shift+525))
        self.coupling.bind(on_text_validate=self.set_coupling_constant)
        self.add_widget(self.coupling)
        
        self.coupling_unit = Label(text='MHz', size=(10, 5), pos=(x_shift-95, y_shift+40), font_size='15sp')
        self.add_widget(self.coupling_unit)
        
        # Asymmetry parameter
        self.asymmetry_label = Label(text='Asymmetry parameter', size=(10, 5), pos=(x_shift+25, y_shift+40), font_size='15sp')
        self.add_widget(self.asymmetry_label)
        
        self.asymmetry = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+515, y_shift+525))
        self.asymmetry.bind(on_text_validate=self.set_asymmetry)
        self.add_widget(self.asymmetry)
        
        # Euler angles
        self.alpha_q_label = Label(text='\N{GREEK SMALL LETTER ALPHA}Q', size=(10, 5), pos=(x_shift-341, y_shift-10), font_size='15sp')
        self.add_widget(self.alpha_q_label)
        
        self.alpha_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+80, y_shift+475))
        self.alpha_q.bind(on_text_validate=self.set_alpha_q)
        self.add_widget(self.alpha_q)
        
        self.alpha_q_unit = Label(text='°', size=(10, 5), pos=(x_shift-250, y_shift-10), font_size='15sp')
        self.add_widget(self.alpha_q_unit)
        
        self.beta_q_label = Label(text='\N{GREEK SMALL LETTER BETA}Q', size=(10, 5), pos=(x_shift-180, y_shift-10), font_size='15sp')
        self.add_widget(self.beta_q_label)
        
        self.beta_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+241, y_shift+475))
        self.beta_q.bind(on_text_validate=self.set_beta_q)
        self.add_widget(self.beta_q)
        
        self.beta_q_unit = Label(text='°', size=(10, 5), pos=(x_shift-89, y_shift-10), font_size='15sp')
        self.add_widget(self.beta_q_unit)
        
        self.gamma_q_label = Label(text='\N{GREEK SMALL LETTER GAMMA}Q', size=(10, 5), pos=(x_shift-19, y_shift-10), font_size='15sp')
        self.add_widget(self.gamma_q_label)
        
        self.gamma_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift+402, y_shift+475))
        self.gamma_q.bind(on_text_validate=self.set_gamma_q)
        self.add_widget(self.gamma_q)
       
        self.gamma_q_unit = Label(text='°', size=(10, 5), pos=(x_shift+73, y_shift-10), font_size='15sp')
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
        
        self.temperature = TextInput(multiline=False, disabled=True, size_hint_x=None, width=65, size_hint_y=None, height=32.5)
        self.temperature.bind(on_text_validate=self.set_temperature)
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
        
        self.set_up_system = Button(text='Set up the system', font_size='16sp', size_hint_y=None, height=40, size_hint_x=None, width=200, pos=(535, 25))
        self.set_up_system.bind(on_press=self.build_system)
        self.add_widget(self.set_up_system)
        

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