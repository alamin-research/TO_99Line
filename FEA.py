# The purpose of this file is to contain all of the functions used for FEA operations

import numpy as np
import scipy

## Construct Global Stiffness Matrices



## Constitutive Matrix

def isotropic2D_plane_stress_constitutive_matrix(E,nu):
    E_matrix = E*(1-nu**2) * np.array([[1, nu, 0],
                                      [nu, 1, 0],
                                      [0, 0, (1-nu)/2]])
    return E_matrix

def isotropic2D_plane_strain_constitutive_matrix(E,nu):
    E_matrix = E/((1+nu)*(1-2*nu)) * np.array([[1-nu, nu, 0],
                                               [nu, 1-nu, 0],
                                               [0, 0, (1-2*nu)/2]])
    return E_matrix

## Integration Methods

def q4_element_gaussian_quadrature_isoparametric_integration_2points(node_coords, constitutive_matrix, thickness=1):
    
    # Taken from Concepts and Applications of Finite Element Analysis 4th Ed by Cook et. al.

    # Initialize the stiffness matrix for this element
    k_el = np.zeros((8,8))

    # Perform the integration at all 4 points
    point = 1/(3**0.5)

    for i in [-point, point]:
        for j in [-point, point]:

            jacobian_det, B = calc_q4_B_matrix(i,j,node_coords)

            if jacobian_det <= 0:
                print("NEGATIVE OR ZERO JACOBIAN!")
            # else:
            #     print(f"det_j at ({i},{j}): {jacobian_det}")
            
            # add the integration point value to the element stiffness
            k_el += np.dot(np.transpose(B), np.dot(constitutive_matrix,B)) * jacobian_det

    # print(f"k_el =\n{k_el}")
    # input()  # This is correct up until this point

    return k_el

## Global Stiffness Functions

def global_stiffness_2d_variable_density_as_csr(element_densities, k_el_function,element_nodes, node_coordinates, constitutive_matrix, penalization_exponent=3):
    
    num_nodes = node_coordinates.shape[0]
    num_ele = element_nodes.shape[0]
    dof_per_node = 2

    # Create the empty global stiffness matrix
    k_global_stiffness_matrix_lil = scipy.sparse.lil_matrix((num_nodes*dof_per_node, num_nodes*dof_per_node))  # Creates a dof x dof zero matrix
    
    # Assemble the global stiffness matrix
    for ele in range(num_ele):

        # Get the nodes for this element and their coordinates
        ele_nodes = element_nodes[ele, :]
        ele_node_coords = node_coordinates[ele_nodes.flatten(),:]

        # get the elemental stiffness matrix
        k_ele = (k_el_function(ele_node_coords,constitutive_matrix)) * (element_densities[ele] ** penalization_exponent)

        # get the relevent dofs and add the elemental stiffness to the global
        nodal_dofs = []
        for node in ele_nodes:
            for i in range(dof_per_node):
                nodal_dofs.append(node*dof_per_node+i)


        # Test values to check assignment
        #print(f"Nodal dofs are:\n{nodal_dofs}")

        
        # add the element stiffness to the global
        # TODO optimize this        
        for i in range(len(nodal_dofs)):
            for j in range(len(nodal_dofs)):
                k_global_stiffness_matrix_lil[nodal_dofs[i],nodal_dofs[j]] += k_ele[i,j]

                
    # For efficiency, modify the type of sparse matrix now that it is assembled
    k_global_stiffness_matrix_csr = k_global_stiffness_matrix_lil.tocsr()

    return k_global_stiffness_matrix_csr


# Solution functions

def solve_unknown_displacements_forces(global_k, fixed_dofs, free_dofs, displacements, loads):

    reactions = np.zeros_like(loads)
    # Partition global_k based on fixed and free dofs
    # Notation based on Concepts and Applications of Finite Element Analysis 4th Ed by Cook et. al. pg 40
    k_11 = global_k[free_dofs,:][:,free_dofs]
    k_12 = global_k[free_dofs,:][:,fixed_dofs]
    k_21 = global_k[fixed_dofs,:][:,free_dofs]
    k_22 = global_k[fixed_dofs,:][:,fixed_dofs]

    displacements[free_dofs,:] = scipy.sparse.linalg.spsolve(k_11, (loads[free_dofs,:] - (k_12 @ displacements[fixed_dofs,:]))).reshape(-1,1)

    reactions[fixed_dofs,:] = (k_21 @ displacements[free_dofs,:]) + (k_22 @ displacements[fixed_dofs,:])

    # print("Load applied at DOFs:", np.where(np.abs(loads) > 1e-12)[0])
    # print("Reaction forces at constrained DOFs:",np.where(np.abs(reactions) > 1e-12)[0])
    # print("Fixed dofs:", sorted(fixed_dofs)[:10], "...")
    # print("Free dofs:", sorted(free_dofs)[:10], "...")

    return displacements, reactions


## Calculate B Matrix
def calc_q4_B_matrix(xi, eta, node_coords):

    # Derivation taken from Cook pg207-208, Eq 6.2-9 to 6.2-12 and compared to pg 98 eq 3.6-6
    # x^ implies x is a vector, [A] implies A is a matrix
    """
    strain = B @ displacement from pg98

    strain = selector @ d/dx (u^)
    selector = [[1, 0, 0, 0],[0, 0, 0, 1], [0, 1, 1, 0]]
    d/dx (u^) = [du/dx, du/dy, dv/dx, dv/dy]  where d is the partial

    d/dx (u^) = [[[gamma],[0]]
                 [[0],[gamma]]] @ d/dxi (u^)
    [gamma] = Jacobian^-1
    d/dxi (u^) = [du/dxi, du/deta, dv/dxi, dv/deta]  where d is the partial

    d/dxi (u^) = [dN_full] @ d^
    dN_full = dN_full = 0.25 * np.array([[N_1xi, 0, N_2xi, 0, N_3xi, 0, N_4xi, 0],
                               [N_1et, 0, N_2et, 0, N_3et, 0, N_4et, 0],
                               [0, N_1xi, 0, N_2xi, 0, N_3xi, 0, N_4xi],
                               [0, N_1et, 0, N_2et, 0, N_3et, 0, N_4et]]) See Cook Eq 6.2-11 or below Ns are described on 6.2-3
    """

    # Calculate dN as [N1, N2, N3, N4]^T * [d/d_xi, d/d_eta]
    dN = 0.25 * np.array([[-(1-eta), (1-eta), (1+eta), -(1+eta)],
                    [-(1-xi), -(1+xi), (1+xi), (1-xi)]])
    
    # jacobian calcs
    jacobian_matrix = np.dot(dN,node_coords)
    jacobian_det = np.linalg.det(jacobian_matrix)
    jacobian_inverse = np.linalg.inv(jacobian_matrix)

    # intermediate calc to find B matrix
    sub_B = np.dot(jacobian_inverse,dN)

    # Testing difference in derivation
    B_old = np.array([[sub_B[0,0],0,sub_B[0,1], 0, sub_B[0,2], 0, sub_B[0,3], 0],
                    [0, sub_B[1,0],0,sub_B[1,1], 0, sub_B[1,2], 0, sub_B[1,3]],
                    [sub_B[1,0],sub_B[0,0],sub_B[1,1], sub_B[0,1], sub_B[1,2], sub_B[0,2], sub_B[1,3], sub_B[0,3]]])
    
    # Below was used to fix the issue with dN having eta and xi flipped
    # selector = np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 1, 1, 0]])
    # gamma2 = scipy.linalg.block_diag(jacobian_inverse,jacobian_inverse)
    # dN_full = 0.25 * np.array([[-(1-eta), 0, (1-eta), 0, (1+eta), 0, -(1+eta), 0],
    #                            [-(1-xi), 0, -(1+xi), 0, (1+xi), 0, (1-xi), 0],
    #                            [0, -(1-eta), 0, (1-eta), 0, (1+eta), 0, -(1+eta)],
    #                            [0, -(1-xi), 0, -(1+xi), 0, (1+xi), 0, (1-xi)]])
    
    # B_new = selector @ gamma2 @ dN_full

    # difference = B_old-B_new

    # if np.any(difference) != 0:
    #     print(f"B_old is:\n{B_old}\nB_new is:\n{B_new}\nThe difference is:\n{difference}")
    #     input()



    # Return B
    return jacobian_det, B_old


## Sensitivity Functions

def strain_energy_gradient_with_respect_to_2D_q4_ele_density(element_nodes, nodal_displacements, node_coordinates, element_densities, k_ele_function, constitutive_matrix, penalization_exponent=3):
    #print("There is currently something wrong with calculating the gradient wrt density")
    
    num_ele = element_nodes.shape[0]
    dof_per_node = 2 # because 2D

    # initialize gradient WithRespectTo
    gradient_wrt_density = np.zeros((num_ele,1))

    # Go through and calculate the gradient at each element
    for ele in range(num_ele):

        # Get the nodes for this element and their coordinates
        ele_nodes = element_nodes[ele, :]
        ele_node_coords = node_coordinates[ele_nodes.flatten(),:]

        # get the elemental stiffness matrix
        k_ele = (k_ele_function(ele_node_coords,constitutive_matrix)) * (element_densities[ele] ** penalization_exponent)

        # get the relevent dofs and add the elemental stiffness to the global
        nodal_dofs = []
        for node in ele_nodes:
            for i in range(dof_per_node):
                nodal_dofs.append(node*dof_per_node+i)

        this_ele_displacements = nodal_displacements[nodal_dofs]

        gradient_wrt_density[ele] = -penalization_exponent * (element_densities[ele] ** penalization_exponent-1) * np.dot(np.transpose(this_ele_displacements), np.dot(k_ele, this_ele_displacements))

    return gradient_wrt_density



# Testing
if __name__ == "__main__":
    import scipy
    test_sparse = scipy.sparse.rand(5,5,density=0.75,format='csr')
    print(test_sparse.toarray())
    selection_list = [3,0,2]
    selection_matrix = test_sparse[selection_list,:][:,selection_list]
    print(selection_matrix.toarray())