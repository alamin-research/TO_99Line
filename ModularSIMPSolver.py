import numpy as np
import scipy
import matplotlib.pyplot as plt

import time

# Import Created Libraries of functions
import GetBoundaryConditions
import Plotter
import SIMPEndConditions
import FEA

# See both the ReadMe and the full explanation at https://github.com/alamin-research/TO_99Line


# Modular SIMP Solver
if __name__ == "__main__":
        
    # 1.  Get the nodal information using method of choice as an np array
    nelx = 100
    nely = 50
    element_nodes, node_coordinates = GetBoundaryConditions.generate_2D_unit_cell_node_grid(nelx=nelx, nely=nely)
    num_ele = element_nodes.shape[0]
    num_nodes = node_coordinates.shape[0]
    
    # 2.  Define the design variables as a numpy array of columns where each row
    # is a variable and the column relates to the the element
    design_variables_per_element = 1
    design_variables = np.ones((design_variables_per_element,num_ele))
    # Set the initial values for each design variable
    vol_frac = 0.5
    design_variables[:,0] = vol_frac  # Set the Inital Volume fraction
    
    # 3. Set Boundary Conditions
    # Manual definition below
    dof_per_node = node_coordinates.shape[1]
    fixed_dofs = []
    fixed_dofs = GetBoundaryConditions.add_rolling_edge_fixed_in_y_to_2D_rectangular_mesh(current_fixed=fixed_dofs, edge_x_position=0, node_coordinates=node_coordinates)            
    fixed_dofs = GetBoundaryConditions.add_fixed_dof_to_node_near_position_to_2D_rectangular_mesh(current_fixed=fixed_dofs, node_x_position=nelx, node_y_position=0, radius=0, node_coordinates=node_coordinates, fix_in_y=False)
    # Set free_dofs based on above
    free_dofs = []
    for i in range(dof_per_node*num_nodes):
        if i not in fixed_dofs:
            free_dofs.append(i)

    # Either load or set the displacement vector and load vector
    nodal_displacements = np.zeros((num_nodes*dof_per_node,1))
    nodal_forces = np.zeros((num_nodes*dof_per_node,1))
    
    
    # 4. Material Properties
    # Main material
    E1 = 1e9
    nu1 = 0.3
    constitutive_matrix = FEA.isotropic2D_plane_stress_constitutive_matrix(E1,nu1)


    # 5. Set the inter-loop variables
    # Set the end conditions for the while loop
    end_condition_density_change = 0.001
    #end_condition_angle_change = 0.5 * np.pi / 180 # 0.5 degree
    
    # Set a variable to control whether iterations continue and set it to change
    # within the loop when all conditions are met
    iteration_count = 0
    max_iterations = 200
    all_end_conditions_met = False
    max_iterations_met = False

    penalization_exponent = 3
    
    while (not all_end_conditions_met) and (not max_iterations_met):
        iteration_count+=1
        
        
        # Store the old design variables
        old_design_variables = np.copy(design_variables)

        # Create the empty global stiffness matrix
        k_global_stiffness_matrix_lil = scipy.sparse.lil_matrix((num_nodes*dof_per_node, num_nodes*dof_per_node))  # Creates a dof x dof zero matrix
        
        # Assemble the global stiffness matrix
        for ele in range(num_ele):

            # Generate the unmodified Constitutive Matrix
            constitutive_matrix_1 = FEA.isotropic2D_plane_stress_constitutive_matrix(E1,nu1)
            # TODO make a list of constitutive matrices and iterate through them for multiple materials

            # Get the nodes for this element and their coordinates
            ele_nodes = element_nodes[ele, :]
            ele_node_coords = node_coordinates[ele_nodes.flatten(),:]

            # get the elemental stiffness matrix
            k_ele = FEA.q4_element_gaussian_quadrature_isoparametric_integration_2points(ele_node_coords,constitutive_matrix_1)

            # get the relevent dofs and add the elemental stiffness to the global
            nodal_dofs = []
            for node in ele_nodes:
                for i in range(dof_per_node):
                    nodal_dofs.append(node*dof_per_node+i)

            # add the element stiffness to the global
            # TODO optimize this
            for i in range(len(nodal_dofs)):
                for j in range(len(nodal_dofs)):
                    k_global_stiffness_matrix_lil[nodal_dofs[i],nodal_dofs[j]] += k_ele[i,j]

        # For efficiency, modify the type of sparse matrix now that it is assembled
        k_global_stiffness_matrix_csr = k_global_stiffness_matrix_lil.tocsr()

        # Solve for displacements and forces
        # forces may not be needed?
        nodal_displacements, nodal_forces = FEA.solve_unknown_displacements_forces(k_global_stiffness_matrix_csr,fixed_dofs,free_dofs,nodal_displacements,nodal_forces)

        # Calculate the gradient for each 




        

        





            


            

        # Solve for the displacements and forces

        # Calculate Objective function and the Sensitivities

        # 

        
        
        # Check to see if end conditions were met
        if iteration_count>=max_iterations:
            max_iterations_met = True
        if (end_condition_density_change > SIMPEndConditions.find_density_max_change(rho=design_variables[:,0], rho_old=old_design_variables[:,0])
            and 
            True):
            all_end_conditions_met = True
        
    
