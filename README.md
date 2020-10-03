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

**Remark:** In the whole program, energies and frequencies are considered quantities with the same physical dimensions. The identification between them is a consequence of setting the Planck constant h (originally the proportionaly constant for the conversion form frequencies to energies) equal to 1.

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
    
Class `Operator` defines the properties of a generic linear application acting in the Hilbert space of a quantum system.
  
Class `Density_Matrix` characterizes the operators which represent the state (pure or mixed) of the quantum system. It is defined by three fundamental properties:
1. hermitianity
1. unit trace
1. positivity

Class `Observable` characterizes the hermitian operators which represent the physical quantities of the system.

### Example of execution

### GUI

### Acknowledgementes







