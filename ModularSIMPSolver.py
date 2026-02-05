import numpy as np
import scipy
import matplotlib.pyplot as plt

import time
import cProfile
import pstats

# Import Created Libraries of functions
import GetBoundaryConditions
import Plotter
import SIMPEndConditions
import FEA
import UpdateDesignVariables

# See both the ReadMe and the full explanation at https://github.com/alamin-research/TO_99Line


# Modular SIMP Solver
def ModSIMPSolver():
    print("ModSIMP attempted")
    mod_SIMP_start_time = time.time()
        
    # 1.  Get the nodal information using method of choice as an np array 
    # where element_nodes has rows for each element, and nodes in each column such that they go counter clockwise left to right
    # and node_coordinates has rows for each node and x,y(,z) for the columns
    nelx = 100
    nely = 50
    element_nodes, node_coordinates = GetBoundaryConditions.generate_2D_unit_cell_node_grid(nelx=nelx, nely=nely)
    num_ele = element_nodes.shape[0]
    num_nodes = node_coordinates.shape[0]
    # precalc the isoparametric shape function derivatives for later use
    dof_per_node = node_coordinates.shape[1]
    dN_cache = FEA.get_2d_dN_cache()
    nodal_dofs_cache, j_det_cache, B_cache = FEA.precache_2d_static_mesh_variables(dof_per_node, dN_cache, node_coordinates, element_nodes)

    # # Test printing values
    # print(f"Element nodes:\n{element_nodes}")
    # print(f"Node Coordinates:\n{node_coordinates}")
    
    # 2.  Define the design variables as a numpy array of columns where each row
    # is a variable and the column relates to the the element
    element_densities = np.ones((num_ele,1))
    element_theta_orientations = np.random.uniform(-np.pi, np.pi, size=element_densities.shape)
    # Set the initial values for each design variable
    vol_frac = 0.5
    element_densities[:,:] = vol_frac  # Set the Inital Volume fraction
    
    # 3. Set Boundary Conditions
    # Manual definition below
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
    nodal_forces[1] = 50
    #nodal_forces[2*(nelx+1)*(nely+1)-2] = 5
    applied_forces = np.copy(nodal_forces)
    
    # Plot boundary conditions
    #Plotter.plot_2D_boundary_conditions(node_coordinates=node_coordinates,fixed_dofs=fixed_dofs,forces=nodal_forces,marker_size=5, max_vector_length=1)
    
    # 4. Material Properties
    # Main material
    E1 = 130e9
    E2 = 10e9
    nu12 = 0.3
    G12 = 4e9
    constitutive_matrix = FEA.orthotropic2D_plane_stress_constitutive_matrix(E1,E2,G12,nu12)
    k_el_function = FEA.q4_element_gaussian_quadrature_isoparametric_integration_2points_full_cache
    ndof = dof_per_node * num_nodes
    

    # 5. Set the inter-loop variables
    # Set the end conditions for the while loop
    end_condition_density_change = 1e-8
    #end_condition_angle_change = 0.5 * np.pi / 180 # 0.5 degree
    
    # Set a variable to control whether iterations continue and set it to change
    # within the loop when all conditions are met
    iteration_count = 0
    max_iterations = 500
    all_end_conditions_met = False
    max_iterations_met = False

    penalization_exponent = 3
    filter_radius = 10
    step_size = 0.01
    max_rotation_per_step = 3 * np.pi/180

    objective_function_history = []
    
    # Calculate filter if necessary
    print("starting filter calc")
    weight_filter = UpdateDesignVariables.calc_2D_filter(element_nodes, node_coordinates, filter_radius)
    
    
    start_of_while_loop_time = time.time()
    print(f"Starting while loop after setup time of {start_of_while_loop_time-mod_SIMP_start_time}")


    while (not all_end_conditions_met) and (not max_iterations_met):

        #print(f"First 10 element orientations in degrees are:\n{np.rad2deg(element_theta_orientations[0:10,0])}")
        this_while_loop_time = time.time()
        iteration_count+=1
        
        # Store the old design variables
        old_element_densities = np.copy(element_densities)
        old_element_theta = np.copy(element_theta_orientations)

        #print(f"Design variables are: {old_design_variables}")

        # Get the global stiffness matrix
        #k_calc_start_time = time.time()
        k_global = FEA.global_stiffness_2d_variable_density_theta_as_csr_cached(nodal_dofs_cache, ndof, num_ele, element_densities, element_theta_orientations, k_el_function, constitutive_matrix, j_det_cache, B_cache, dN_cache, penalization_exponent=3)
        #k_calc_time = time.time() - k_calc_start_time
        # print("K global found")

        # Check values
        #Plotter.plot_global_stiffness_sensitivities(k_global, node_coordinates, scale=10)
        #print(k_global.toarray())
        #print(f"Symmetry Check: {k_global.toarray() - k_global.T.toarray()}")

        # Solve for displacements and forces
        # forces may not be needed?
        #FEA_solve_start_time = time.time()
        nodal_displacements, nodal_forces = FEA.solve_unknown_displacements_forces(k_global,fixed_dofs,free_dofs,nodal_displacements,applied_forces)
        #FEA_solve_time = time.time() - FEA_solve_start_time
        # Optionally show the displacements
        #Plotter.plot_2D_displacments(node_coordinates, nodal_displacements, exaggeration=10e5)

        # Calculate Objective function
        objective_function_history.append((nodal_displacements.T @ (k_global.dot(nodal_displacements)))[0,0])

        # Calculate the gradient WithRespectTo each variable
        #find_density_gradient_start_time = time.time()
        gradient_wrt__density, gradient_wrt_theta = FEA.strain_energy_gradient_with_respect_to_2D_q4_ele_density_theta_full_cache(node_coordinates, nodal_displacements, element_nodes, element_densities, element_theta_orientations, k_el_function, constitutive_matrix,j_det_cache, B_cache, dN_cache,penalization_exponent=3)
        #find_density_gradient_time = time.time() - find_density_gradient_start_time
        #Plotter.plot_2D_weight_gradient(element_nodes,node_coordinates,fixed_dofs,gradient_in=gradient_wrt__density,iteration_num=iteration_count)
        #print(gradient_wrt__density)
        
        # Update the design variables
        #update_start_time = time.time()
        element_densities = UpdateDesignVariables.simple_2D_grid_density_variable_update_with_filter(element_densities,gradient_wrt__density, step_size=step_size, volfrac=vol_frac, weight_filter=weight_filter,ele_num=num_ele)        # design_variables[0,:] = design_variables[0,:] + 100*(gradient_wrt__density).reshape(1,-1)
        element_theta_orientations = UpdateDesignVariables.simple_2D_grid_theta_variable_update_with_filter(element_theta_orientations,gradient_wrt_theta, max_rotation_per_step=max_rotation_per_step, weight_filter=weight_filter,ele_num=num_ele)        # design_variables[0,:] = design_variables[0,:] + 100*(gradient_wrt__density).reshape(1,-1)
        
        #update_time = time.time() - update_start_time
        # element_densities = np.clip(element_densities, 0, 1)

        # print("loop end\n\n\n")
        
        density_figure = Plotter.plot_2D_mesh_densities_thetas(element_nodes, element_theta_orientations, node_coordinates, element_densities, iteration_count,max_line_length=0.5)

        
        # Check to see if end conditions were met
        if iteration_count>=max_iterations and not max_iterations_met:
            max_iterations_met = True
            print("Max iterations met")
            #print(f"final volfrac = {np.average(element_densities)}")
        if (end_condition_density_change > SIMPEndConditions.find_density_max_change(rho=element_densities, rho_old=old_element_densities)
            and 
            True):
            all_end_conditions_met = True
            print("End condition satisfied")

        
        print(f"Iteration {iteration_count}: Strain energy = {objective_function_history[-1]} . Loop = {time.time() - this_while_loop_time}s")
        #print(f"Iteration {iteration_count}: KCalc={k_calc_time} . FEASolve={FEA_solve_time} . Gradient={find_density_gradient_time} . Update={update_time}")
        
    print(f"final volfrac = {np.average(element_densities)}")
    
    plt.figure()
    plt.title("Optimization History")
    plt.xlabel("Iterations")
    plt.ylabel("Strain energy")
    plt.plot(range(len(objective_function_history)),objective_function_history)
    plt.show(block=False)
    plt.pause(10)

    #input()  # Pause to show figures

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    ModSIMPSolver()

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    stats.print_stats(20)
    