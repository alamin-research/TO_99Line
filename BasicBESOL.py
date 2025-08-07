# Implementation of the matlab code esoL.m found in
# Bi-directional Evolutionary Structural Optimization on Advanced Structures and Materials: A Comprehensive Review
# by Liang Xia • Qi Xia • Xiaodong Huang • Yi Min Xie

# This python implementation by Caleb Tanner. August 5, 2025

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

def esoL_python(nelx, nely, volfrac, er, rmin, F, fixeddofs):
    ## Initialization
    vol = 1
    change = 1
    ij = 0 # iteration counter
    x = np.ones((nely,nelx))
    c = []  # compliance per iteration tracker

    ## Material Properties
    E0 = 1
    nu = 0.3

    ## Prepare Finite Element Analysis
    A11 = np.array([[12, 3, -6, -3],
                    [3, 12, 3, 0],
                    [-6, 3, 12, -3],
                    [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3],
                    [-3, -6, -3, -6],
                    [0, -3, -6, 3],
                    [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9],
                    [3, -4, -9, 4],
                    [-2, -9, -4, -3],
                    [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9],
                    [-3, 2, 9, -2],
                    [4, 9, 2, 3],
                    [-9, -2, 3, 2]])
    KE = 1/(1-nu**2)/24 * (np.vstack((np.hstack((A11, A12)),np.hstack((A12,A11)))) +
          nu * np.vstack((np.hstack((B11, B12)),np.hstack((B12,B11)))))
    
    # Create the element-dof relationships reindexed to start at 0
    # nodenrs is not used out of creating edofVec which is similarly
    # not used outside of creating edofMat.
    # For consistency, the elements will still begin counting with the top
    # left element, increment downward and then step one column to the right
    # before incrementing downward again. Nodes increment in the same way
    # edofMat is the the list of dofs in each row where each row refers to
    # the element of the same index beginning with the second node, here indexed 
    # as 1 (since python indexing starts with 0), and continuing ccw
    # Below edofMat is explicitly created
    edofMat = np.ones((nelx*nely,8))*-1 #-1 to check unmodified
    for i in range(nelx):
        for j in range(nely):
            element_number = j + i*nely
            # node numbers:
            bl = j + 1 + i*(nely+1)
            br = bl + (nely+1)
            tr = br - 1
            tl = bl -1
            # get dof numbers from nodes
            edofMat[element_number,:] = np.array([2*bl, 2*bl+1,2*br, 2*br+1,2*tr, 2*tr+1,2*tl, 2*tl+1])

    #print(edofMat)

    # Next get the iK and jK vectors for positioning stiffnesses in the sparse matrix
    iK = np.transpose(np.kron(edofMat,np.ones((8,1)))).reshape(-1,1,order='F') # Reshape the same way matlab does
    jK = np.transpose(np.kron(edofMat,np.ones((1,8)))).reshape(-1,1,order='F') 

    # np.set_printoptions(threshold=np.inf)
    # print(np.hstack((iK, jK)))

    ## Define Loads and Supports
    # made these inputs instead

    U = np.zeros(((2*(nely+1)*(nelx+1)),1)) # displacement vector? Confirm later
    alldofs = list(range(2*(nely+1)*(nelx+1)))
    freedofs = np.setdiff1d(np.array(alldofs), np.array(fixeddofs)).tolist()

    ## Prepare filter
    # Allocate space for the maximum number of elements that can be within rmin of each other
    iH = np.ones((nelx*nely*(2*(math.ceil(rmin)-1)+1)**2,1))
    jH = np.ones_like(iH)
    sH = np.zeros_like(iH)
    k = 0

    for i1 in range(nelx):
        for j1 in range(nely):
            # Get the element number in question
            e1 = (i1)*nely + j1  # Adjusted for 0 indexing
            # check every elemen that might be in range by checking a square around e1
            for i2 in range(max(i1-(math.ceil(rmin)-1),1),min(i1+(math.ceil(rmin)-1),nelx)):
                for j2 in range(max(j1-(math.ceil(rmin)-1),1),min(j1+(math.ceil(rmin)-1),nely)):
                    e2 = (i2)*nely + j2  # adjusted for 0 indexing
                    iH[k] = e1
                    jH[k] = e2
                    k+=1 # moved here for 0 indexing
                    sH[k] = max(0, rmin - math.sqrt((i1-i2)**2 + (j1-j2)**2))

    H = scipy.sparse.coo_array((sH.flatten(), (iH.flatten(), jH.flatten())), shape=(nelx * nely, nelx * nely))
    print(f"The shape of H is {H.shape}")
    Hs = H.sum(axis=1).reshape(-1,1,order='F') # Get the denominator for total contributors to each row which refers to that element

    ## Start iteration
    while change > 0.001:
        ij += 1
        vol = max(vol*(1-er), volfrac)

        # Track last version after first iteration
        if ij > 1:
            olddc = dc
        
        ## FE-Analysis
        # get the values of the stiffness matrix in the same position as iK and jk for the sparse stiffness matrix
        sK =(np.dot(KE.reshape(-1,1,order='F'),np.maximum(1e-9,x.reshape(1,-1,order='F'))) * E0).reshape( 64*nelx*nely,1,order='F')
        K = scipy.sparse.coo_array((sK.flatten(), (iK.flatten().astype(int), jK.flatten().astype(int))))
        K = (K+K.transpose())/2

        print(K[np.ix_(freedofs, freedofs)].shape)
        U[freedofs,:] = scipy.sparse.linalg.spsolve(K[np.ix_(freedofs, freedofs)],F[freedofs,:]).reshape(-1,1)

        # # Test output 
        # print(U) resulted in same as Matlab

        ## Objective function and sensitivity Analysis
        ce = ((((U[edofMat.astype(int)].reshape(-1,8,order='F') @ KE) * U[edofMat.astype(int)].reshape(-1,8,order='F'))).sum(axis=1)).reshape(nely,nelx,order='F') 
        dc = (E0 * x * ce) # This line is flipped with the next 
        c.append(dc.sum())
        
        ## Filtering/Modicication of sensitivities
        # print(f"The shape of filtered dc is {(H @ dc.reshape(-1, 1, order='F')).shape}")
        # print(f"The shape of Hs is {Hs.shape}")
        dc = (H @ dc.reshape(-1, 1, order='F') / Hs).reshape(nely, nelx, order='F')
        if ij > 1:
            dc = (dc + olddc)/2

        ## Print Results and plot densities
        if ij >= 10:
            recent_5 = np.array(c[ij - 5 : ij])     # MATLAB 1-based -> Python 0-based
            prev_5 = np.array(c[ij - 10 : ij - 5])
            change = abs(prev_5.sum() - recent_5.sum()) / recent_5.sum()

        print(f"It.:{ij:3d} Obj.:{c[ij-1]:8.4f} Vol.:{x.mean():4.3f} ch.:{change:4.3f}")

        plt.clf()
        plt.imshow(-x, cmap='gray', aspect='equal')
        plt.axis('off')
        plt.pause(1e-6)

        ## Optimality criteria update of design variables
        l1 = np.min(dc)
        l2 = np.max(dc)
        while (l2-l1)/(l1+l2) > 1e-9:
            th = (l1+l2)/2  # threshold
            x = np.maximum(0,np.sign(dc-th))
            if np.average(x) - vol > 0:
                l1 = th
            else:
                l2 = th

    plt.show()
        

        




if __name__ == "__main__":

    # testing

    # Coordinate system for calculations is y+ down, x+ right

    # Half-MBB Beam
    nelx = 100
    nely = 40
    F_1 = np.zeros((2*(nely+1)*(nelx+1),1))
    F_1[1,0] = -1
    fixeddofs_1 = sorted(set(list(range(0,(2*(nely+1)),2)) + [2*(nely+1)*(nelx+1)-1])) # fixes the midplane wall in the x direction and the y=0, x=max dof in the y direction 

    esoL_python(nelx=nelx, nely=nely, volfrac=0.6, er=0.02, rmin=6, F=F_1, fixeddofs=fixeddofs_1)