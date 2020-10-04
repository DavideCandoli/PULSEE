# NQR-NMR Simulation Software

## Author: Davide Candoli (Università di Bologna)

This project consists of a software which simulates a typical nuclear quadrupole/magnetic resonance experiment on a solid state sample, describing the dynamics of nuclear spins in condensed matter under the effect external magnetic fields and reproducing the traditional results observed in laboratory.

## Physics background

Each atomic nucleus in a single crystal is endowed with an intrinsic angular momentum, named spin, and a corresponding intrinsic magnetic moment. The interaction between this latter and any applied magnetic field provides a means for the manipulation of nuclear spins and the study of the magnetic properties of a sample.

One basic assumption of the simulations implemented by this program is that the spins under study belong to a periodic crystal where the set of nuclei can be treated as an ideal *statistical ensemble*: this allows to describe the whole many-spin system appealing to the *mixed state* of a single spin. Thus, the Hilbert space of the problem reduces to the 2s+1-dimensional space of a single spin, where each state is represented by a mixed density matrix.

Let us consider such a system under the effect of a static magnetic field. In these circumstances, the degeneracy of spin states is broken and they arrange into equally spaced energy levels, with an energy separation determined by the strength of the field and the gyromagnetic ratio of the nuclear spin.

In order to induce transitions between these states, a pulsed electromagnetic wave must have a frequency tuned with the resonance frequency of the system. This is the concept at the basis of the nuclear magnetic resonance (NMR) technique.

In nuclear quadrupole resonance (NQR), what lifts the degeneracy of spin states is the quadrupolar interaction between the charge distribution of the nucleus and the electric field gradient (EFG) due to the surrounding electron cloud. The applied electromagnetic pulses succeed in turning the spin's direction only if their frequency coincides with one of the several resonance frequencies of the system.

In general, experiments deal with intermediate situations between NMR and NQR, where both the interaction with the magnetic field (Zeeman interaction) and with the EFG (quadrupolar interaction) must be taken into account in order to determine the energy spectrum.

In order to extract information about the interactions of nuclear spins and the magnetic properties of the sample, NMR and NQR serve as successful techniques of investigation. Indeed, after the application of the designed pulse sequence, the system's magnetization typically has a non-zero component in the direction of the coil employed for the generation of the pulse. This component changes with time in a way that depends on the microscopic properties of the nuclei, and progressively goes to zero due to the unavoidable dephasing of different spins, which takes place with a characteristic time T<sub>2</sub> called *coherence time*.

The time dependence of the magnetization is measured through the acquisition of the current induced in the coil. Then, performing the Fourier analysis of this signal, one is able to extract a lot of information about the system, according to the position of the peaks of the spectrum, their shape and their sign.

### Unit standard of the software

In order to save processing power and lighten calculations, a suitable choice of units of measure has been taken.

**Remark:** In the whole program, energies and frequencies are considered quantities with the same physical dimensions. The identification between them is a consequence of setting the Planck constant h (originally the proportionaly constant for the conversion from frequencies to energies) equal to 1.

The standard units employed in the software are listed below.

| physical quantity  | unit  |
| ------------------ | ----- |
| gyromagnetic ratio | MHz/T |
|   magnetic field   |   T   |
|  energy/frequency  |  MHz  |
|    temperature     |   K   |
|        time        |   us  |

Angles do not have a standard unit: they are measured in radians when they are passed directly to the software's functions, while they are measured in degrees when they are inserted in the GUI.

## Software

### Prerequisites

The software has been written in Python 3.7.

The operative systems where the code has been tested and executed are
* Ubuntu
* Windows 10 (through the Spyder interface provided by the distribution Anaconda)

The program makes wide use of many of the standard Python modules (namely `numpy`, `scipy`, `pandas`, `matplotlib`) for its general purposes.

Tests have been carried out using the `pytest` framework and the `hypothesis` module.

`pytest` -> https://docs.pytest.org/en/stable/

`hypothesis` -> https://hypothesis.readthedocs.io/en/latest/

The GUI has been implemented with the tools provided by the Python library `kivy`.

`kivy` -> https://kivy.org/#home

In order to run the GUI, it is required the additional installation of the module `kivy.garden` and `garden.matplotlib.backend_kivy`.

`kivy.garden` -> https://kivy.org/doc/stable/api-kivy.garden.html

`garden.matplotlib.backend_kivy` -> https://github.com/kivy-garden/garden.matplotlib/blob/master/backend_kivy.py

### Modules of the software

The program is made up by 6 modules. Each of them is described in full detail in the wiki page of the repository of GitHub which hosts the project:

https://github.com/DavideCandoli/NQR-NMRSimulationSoftware/wiki

Below, the content and usage of these modules is reported briefly:

* `Operators`

  This module, together with `Many_Body`, is to be considered a sort of toolbox for the simulations of generic quantum systems. It contains the definitions of the classes and functions related to the basic mathematical objects which enter the treatment of a quantum system. `Operators` focuses on the properties of a single system, while `Many_Body` extends these features to systems made up of several particles.
  
  The classes and subclasses defined in `Operators` belong to the following inheritance tree
  
  * `Operator`
    * `Density_Matrix(Operator)`
    * `Observable(Operator)`
    
  Class `Operator` defines the properties of a generic linear application acting in the Hilbert space of a finite-dimenstional quantum system. Its main attribute is `matrix`, a square array of complex numbers which gives the matrix representation of the operator in a certain basis. The methods of this class implement the basic algebraic operations and other common actions involving operators, such as the change of basis.
  
  Class `Density_Matrix` characterizes the operators which represent the state (pure or mixed) of the quantum system. It is defined by three fundamental properties:
  1. hermitianity
  1. unit trace
  1. positivity

  Class `Observable` characterizes the hermitian operators which represent the physical quantities of the system.
  
  Other functions defined in this module perform:
  * the calculation of the first few terms of the Magnus expansion of a time-dependent Hamiltonian, which turn useful in the approximation of the evolution operator; 
  * the calculation of the canonical density matrix corresponding to the equilibrium state of a system at the specified temperature.
  
* `Many_Body`

  This module contains two function definitions which allow to pass from a single particle Hilbert space to a many particle space and viceversa.
  
  * `tensor_product_operator`
  
    Takes two operators of arbitrary dimensions and returns their tensor product.
    
  * `partial_trace`
  
    Takes an operator acting on the Hilbert space of a many-particle system and extracts its partial trace over the specified subspace.
    
  **Remark:** These functions have been tested working properly, but up to now no simulation has been carried out which makes use of them. Anyway, new simulations may rely on these functions, so they have been included anyway in the program.
    
* `Nuclear_Spin`

  In this module, the definitions of `Operators` are employed to build up the class which represents a spin of an atomic nucleus.
  
  Class `Nuclear_Spin` is characterized by a quantum number, a gyromagnetic ratio and a set of methods which return the spherical and cartesian components of the spin vector operator.
  
* `Hamiltonians`

  This file is dedicated to the definitions of the functions which return the pieces of the Hamiltonian of a nuclear spin system in an NMR/NQR experiment.
  
  * `h_zeeman`
    
    Builds up the Hamiltonian of the interaction between the spin and an external static magnetic field, after its magnitude and direction has been given.
    
  * `h_quadrupole`
  
    Builds up the Hamiltonian of the interaction between the nuclear electric quadrupole momentum and the EFG, after the coupling constant of the interaction, the asymmetry of the EFG and the direction of its principal axes have been given.
    
    This function calls in turn the functions `v0_EFG`,`v1_EFG`, `v2_EFG` for the computation of the spherical components of the EFG tensor, which enter the expression of the quadrupole Hamiltonian.
    
  * `h_single_mode_pulse`
  
    Returns the Hamiltonian of interaction of a spin with a linearly polarized electromagnetic wave, once the properties of the wave and the time of evaluation have been passed.
    
    This function is called by `h_multiple_mode_pulse`, which returns the Hamiltonian of interaction with a superposition of pulses.
    
    In turn, `h_multiple_mode_pulse` is called inside `h_changed_picture`, which, in the given instant of time, evaluates the full Hamiltonian of the system (comprised of the Zeeman, quadrupole and time-dependent contributions) and returns the same Hamiltonian expressed in a different dynamical picture. This passage is required by the implementation of the evolution of the system, which is described later.
    
### Example of execution

### GUI

### Acknowledgementes







