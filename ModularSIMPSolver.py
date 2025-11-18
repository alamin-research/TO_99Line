import numpy as np
import scipy
import matplotlib.pyplot as plt

import time

# Import Created Libraries of functions
import GetBoundaryConditions
import Plotter
import SIMPEndConditions
import FEA
import UpdateDesignVariables

# See both the ReadMe and the full explanation at https://github.com/alamin-research/TO_99Line


# Modular SIMP Solver
if __name__ == "__main__":
    print("ModSIMP attempted")
        
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
    fixed_dofs = GetBoundaryConditions.add_fixed_dof_to_node_near_position_to_2D_rectangular_mesh(current_fixed=fixed_dofs, node_x_position=nelx, node_y_position=0, radius=0, node_coordinates=node_coordinates, fix_in_x=False)
    fixed_dofs = GetBoundaryConditions.add_rolling_edge_fixed_in_y_to_2D_rectangular_mesh(current_fixed=fixed_dofs, edge_x_position=0, node_coordinates=node_coordinates)            
    
    # Set free_dofs based on above
    free_dofs = []
    for i in range(dof_per_node*num_nodes):
        if i not in fixed_dofs:
            free_dofs.append(i)

    # Either load or set the displacement vector and load vector
    nodal_displacements = np.zeros((num_nodes*dof_per_node,1))
    nodal_forces = np.zeros((num_nodes*dof_per_node,1))
    nodal_forces[1] = -50
    applied_forces = np.copy(nodal_forces)
    
    # Plot boundary conditions
    Plotter.plot_2D_boundary_conditions(node_coordinates=node_coordinates,fixed_dofs=fixed_dofs,forces=nodal_forces,marker_size=3, max_vector_length=5)
    
    # 4. Material Properties
    # Main material
    E1 = 1e9
    nu1 = 0.3
    constitutive_matrix = FEA.isotropic2D_plane_stress_constitutive_matrix(E1,nu1)


    # 5. Set the inter-loop variables
    # Set the end conditions for the while loop
    end_condition_density_change = 0.00001
    #end_condition_angle_change = 0.5 * np.pi / 180 # 0.5 degree
    
    # Set a variable to control whether iterations continue and set it to change
    # within the loop when all conditions are met
    iteration_count = 0
    max_iterations = 5
    all_end_conditions_met = False
    max_iterations_met = False

    penalization_exponent = 3
    filter_radius = 10
    step_size = 0.1

    objective_function_history = []
    
    # Calculate filter if necessary
    print("starting filter calc")
    weight_filter = UpdateDesignVariables.calc_2D_filter(element_nodes, node_coordinates, filter_radius)
    
    print("Starting while loop")

    while (not all_end_conditions_met) and (not max_iterations_met):
        iteration_count+=1
        
        
        # Store the old design variables
        old_design_variables = np.copy(design_variables)

        # Get the global stiffness matrix
        k_global = FEA.global_stiffness_2d_variable_density_as_csr(element_densities=design_variables[0,:].reshape(-1,1),k_el_function=FEA.q4_element_gaussian_quadrature_isoparametric_integration_2points,element_nodes=element_nodes,node_coordinates=node_coordinates,constitutive_matrix=constitutive_matrix,penalization_exponent=3)
        # print("K global found")
        # Solve for displacements and forces
        # forces may not be needed?
        nodal_displacements, nodal_forces = FEA.solve_unknown_displacements_forces(k_global,fixed_dofs,free_dofs,nodal_displacements,applied_forces)
        # Optionally show the displacements
        Plotter.plot_2D_displacments(node_coordinates, nodal_displacements, exaggeration=10e5)
        input()

        # Calculate Objective function
        objective_function_history.append((nodal_displacements.T @ (k_global.dot(nodal_displacements)))[0,0])
        print(f"Iteration {iteration_count}: Strain energy = {objective_function_history[-1]}")

        # Calculate the gradient WithRespectTo each variable
        gradient_wrt__density = FEA.strain_energy_gradient_with_respect_to_2D_q4_ele_density(element_nodes,nodal_displacements,node_coordinates,design_variables[0,:].reshape(-1,1),FEA.q4_element_gaussian_quadrature_isoparametric_integration_2points,constitutive_matrix,penalization_exponent=3)
        Plotter.plot_2D_weight_gradient(element_nodes,node_coordinates,fixed_dofs,gradient_in=gradient_wrt__density,iteration_num=iteration_count)
        print(gradient_wrt__density)
        input()
        # Update the design variables
        design_variables[0,:] = UpdateDesignVariables.simple_2D_grid_density_variable_update_with_filter(design_variables[0,:],gradient_wrt__density, step_size=step_size, volfrac=vol_frac, weight_filter=weight_filter,ele_num=num_ele)        # design_variables[0,:] = design_variables[0,:] + 100*(gradient_wrt__density).reshape(1,-1)
        # design_variables[0, :] = np.clip(design_variables[0, :], 0, 1)

        # print("loop end\n\n\n")

        
        # Check to see if end conditions were met
        if iteration_count>=max_iterations:
            max_iterations_met = True
            print("Max iterations met")
            print(f"final volfrac = {np.average(design_variables[0,:])}")
        if (end_condition_density_change > SIMPEndConditions.find_density_max_change(rho=design_variables[:,0], rho_old=old_design_variables[:,0])
            and 
            True):
            all_end_conditions_met = True
            print("End condition satisfied")
        
    
    plt.figure()
    plt.title("Optimization History")
    plt.xlabel("Iterations")
    plt.ylabel("Strain energy")
    plt.plot(range(len(objective_function_history)),objective_function_history)
    plt.show(block=False)
    plt.pause(10)