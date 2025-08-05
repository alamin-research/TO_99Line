# Modular Topology Optimization Library

# ReadMe
The point of this Library is to create a structure that can be used in topology optimization research and is as modular as possible for prototyping and AI research. There are several solvers that can accept a variety of 2D and 3D functions from the available packages. Below should be a full description of the different functions available including their inputs and how they should be used.

## Acronyms Used
Listed alphabetically:
* dofs: Degrees of Freedom
* E: Youngs Modulus, a material property
* np: the numpy package available in python used frequently in linear algebra
* nu: Poisson's ratio, a material property
* SIMP: Solid Isotropic Material with Penalization. A method of topology optimization


# Modular SIMP Solver
The Modular Simp Solver is just a framework for running the available methods in a structured way. In order for the flow to work the following steps need done in order.

1. Define the Mesh information
    * element_nodes: The element nodes as a np array where each row refers to an element of the same index as the row and the elements in that row are the indices of the nodes that define the row's element
    * node_coordiantes: The node coordinates as an np array where each row refers to a node of the same index as the row and the elements in that row are the positional coordinates of that node in x,y notation when in 2D and x, y, z in 3D
    * num_ele: The number of elements should always match the number of rows in element_nodes
    * num_nodes: The number of nodes should always match the number of rows in node_coordinates

2. Define the design variables
    * design_variables_per_element: Set the number of design variables per element. For standard SIMP, volume/density is the only design parameter. For volume+orientaion in 2D it would be 2 (density and orientation defined with 1 angle), and in 3D it would be 3 (density and orientation defined with 2 angles). 
    * design_variables: The variables that will be optimized throughout the program. For flattening purposes this is done where each row refers to a type of variable (density or angle) and the column is the element to which that variable is applied. Other variables may be set to initiliaze design variables, but only design variables should be relied on past this step

3. Set the boundary conditions
    * fixed_dofs: get a python style list that contains every degree of freedom that is fixed. The number per node should be found based on the number of columns in node_coordinates. Use methods from the GetFixedNodes package to either load a set of fixed nodes or add them based on manual selection. **Manual selection is only recommended for regular grid meshes**
    * free_dofs: should be every dof that is not in fixed_dofs. Nothing should be added to fixed_dofs after the assignment of free_dofs

4. Set the Material Properties
    * E#, nu#: 

5. Set the End conditions
    * There are several options for end conditions that are available for the user. Checking the available package labeled SIMPEndCondtions or the documentation below for available options, but it is recommended that their thresholds be set here before main while loop. Always having a max iterations limit is also recommended.
    * end_condition_density_change: Set a value for the change in density that triggers the end condition. For example adding the condition  "if: end_condition_density_change > SIMPEndConditions.find_density_max_change(...)" would trigger only if the maximum element-wise change was less than the set value
    * end_condition_angle_change: Same as above but used for all angle design parameters
    * max_iterations: the maximum number of optimization loops that will be performed before terminating


# FEA

The general goal of linear 