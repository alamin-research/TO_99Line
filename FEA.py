# The purpose of this file is to contain all of the functions used for FEA operations

import numpy as np

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

            B = calc_q4_B_matrix(i,j,node_coords)
            
            # add the integration point value to the element stiffness
            k_el += np.dot(np.transpose(B), np.dot(constitutive_matrix,B))

    return k_el


# Solution functions

def solve_unknown_displacements_forces(global_k, fixed_dofs, free_dofs, displacements, loads):

    # Partition global_k based on fixed and free dofs
    # Notation based on Concepts and Applications of Finite Element Analysis 4th Ed by Cook et. al. pg 40
    k_11 = global_k[free_dofs,:][:,free_dofs]
    k_12 = global_k[free_dofs,:][:,fixed_dofs]
    k_21 = global_k[fixed_dofs,:][:,free_dofs]
    k_22 = global_k[fixed_dofs,:][:,fixed_dofs]

    displacements[free_dofs,:] = scipy.sparse.linalg.spsolve(k_11, (displacements[free_dofs,:] - (k_12 @ displacements[fixed_dofs,:])))

    loads[fixed_dofs,:] = (k_21 @ displacements[free_dofs,:]) + (k_22 @ displacements[fixed_dofs,:])

    return displacements, loads


## Calculate B Matrix
def calc_q4_B_matrix(xi, eta, node_coords):
    # Calculate dN as [N1, N2, N3, N4]^T * [d/d_xi, d/d_eta]
    dN = np.array([[-(1-xi), (1-xi), (1+xi), -(1+xi)],
                    [-(1-eta), -(1+eta), (1+eta), (1-eta)]])
    
    # jacobian calcs
    jacobian_matrix = 0.25 * np.dot(dN,node_coords)
    jacobian_det = np.linalg.det(jacobian_matrix)
    jacobian_inverse = np.linalg.inv(jacobian_matrix)

    # intermediate calc to find B matrix
    sub_B = np.dot(jacobian_inverse,dN)

    # Return B
    return np.array([[sub_B[0,0],0,sub_B[0,1], 0, sub_B[0,2], 0, sub_B[0,3], 0],
                    [0, sub_B[1,0],0,sub_B[1,1], 0, sub_B[1,2], 0, sub_B[1,3]],
                    [sub_B[1,0],sub_B[0,0],sub_B[1,1], sub_B[0,1], sub_B[1,2], sub_B[0,2], sub_B[1,3], sub_B[0,3]]])


# Testing
if __name__ == "__main__":
    import scipy
    test_sparse = scipy.sparse.rand(5,5,density=0.75,format='csr')
    print(test_sparse.toarray())
    selection_list = [3,0,2]
    selection_matrix = test_sparse[selection_list,:][:,selection_list]
    print(selection_matrix.toarray())