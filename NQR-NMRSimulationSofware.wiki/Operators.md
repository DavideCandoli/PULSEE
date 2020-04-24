# Operators

Operators is the package which provides the classes for the representation of operators in finite-dimensional Hilbert spaces (as is the case of spin systems). Currently, these classes are:
* `Operator`
* `Density_Matrix(Operator)`
* `Observable(Operator)`
`Operator` is to be intended as the mathematical object without any physical meaning, while the instances of its subclasses `Density_Matrix` and `Observable` embody the physical objects indicated by their names: respectively, the state of the system and its measurable properties.
All of them possess an attribute called `matrix` which expresses the matrix representation of the operator in the assumed basis set.

### Class Operator

#### Methods
1. **`Operator(x)`**
   Constructs an instance of `Operator`.
   ##### Parameters
   * `x`: Either`int` or `ndarray`. In the latter case, the constructor checks that the given object is a square array, and raises appropriate errors if any of these conditions fail.
   ##### Action
   When `x` is an integer, the constructor initialises `matrix` as an `x`-dimensional identity array. When `x` is an array, it is assigned directly to `matrix`.
   ##### Returns
   The initialised Operator object.