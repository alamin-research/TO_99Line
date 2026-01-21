

import numpy as np
import scipy

def calc_2D_filter(element_nodes, node_coordinates, radius):

    """ Template
    # For every element
    for i in range(nelx):
        for j in range(nely):
            # Initialize a tracker called W sum that tracks the total weight of nearby elements
            w_sum = 0 
            # max available index changed to 0 to match different indices in python
            for k in range(max(i-round(rmin),0),min(i+round(rmin),nelx)):
                for l in range(max(j-round(rmin),0),min(j+round(rmin),nely)):
                    # find the distance between this element and every other element that could be nearby. The complicated for loops above pre exclude elements that are too far away
                    fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                    w_sum +=max(0,fac)
                    dcn[j,i] += max(0,fac)*x[l,k]*dc[l,k]
            dcn[j,i] = dcn[j,i]/(x[j,i]*w_sum)
            if (x[j,i]*w_sum) == 0:
                print("There is a 0 in the denominator")
                print(f"X value at j={j}, i={i} is {x[j,i]}")
                print(f"W_sum is {w_sum}")
                input()
    """


    # Find the element centers
    ele_num = element_nodes.shape[0]
    ele_centroids = np.zeros((ele_num,2))
    for ele in range(ele_num):
        this_ele_coords = node_coordinates[list(element_nodes[ele,:]),:]
        ele_centroids[ele,:] = (this_ele_coords.sum(axis=0))/this_ele_coords.shape[0]
    print("Centers found")
    # Filter the gradient

    # ChatGPT assisted weight filter calculation
    tree = scipy.spatial.cKDTree(ele_centroids)

    # initialize the sparse matrix for the filter
    weight_filter = scipy.sparse.lil_matrix((ele_num,ele_num), dtype=np.float64)

    for ele in range(ele_num):
        # get the indices of all elements within rmin of this one
        idx = tree.query_ball_point(ele_centroids[ele,:],r=radius)

        if len(idx) == 0:
            continue # skip if nothing is in range

        # Distances to neighbors in range
        distances = np.linalg.norm(ele_centroids[idx] - ele_centroids[ele], axis=1)

        weights = np.maximum(0.0, radius - distances)

        weights /= weights.sum()

        weight_filter[ele,idx] = weights

    return weight_filter.tocsr()

def simple_2D_grid_density_variable_update_with_filter(densities, gradient, step_size, volfrac, weight_filter, ele_num):
    
    
    # Filter the gradient

    filtered_gradient = weight_filter @ gradient

    """Template
    # Scale the weights according the the current weights and gradient so the volfrace is reached
    l1 = 0 
    l2 = 100000
    move = 0.2
    
    # print(f"Max dc is {np.max(dc)}, min of dc is {np.min(dc)}")
    xnew = 2*nelx*nely
    while abs(np.sum(xnew)/(nelx*nely) -volfrac) > 1e-8:
        lmid = (l2+l1)/2 
        
        # Splitting up xnew for readability and trouble shooting
        # print(f"xnew1 arg 1 is {x+move}")
        # print(f"xnew1 arg 2 is {(-dc*(1/lmid)**0.5)}")
        xnew1 = np.minimum(x+move, x * (-dc*(1/lmid))**0.5)
        xnew2 = np.minimum(1,xnew1)
        xnew3 = np.maximum(x-move,xnew2)
        xnew = np.maximum(0.000000001,xnew3)
        
        if (np.sum(xnew)) - volfrac*nelx*nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    # print(f"L1 = {l1}, L2 = {l2}")
    # print(f"This volume fraction is {np.sum(xnew)/(nelx*nely)}")
    
    return xnew
    """
    # Adjust the densities
    l1 = 0 
    l2 = 1e15

    # Guarantee that the while loop runs at least once
    while True:
        lmid = (l2+l1)/2 
        
        # Splitting up xnew for readability and trouble shooting
        xnew1 = np.minimum(densities+step_size, densities * (filtered_gradient*(1/lmid)**0.5))
        xnew2 = np.minimum(1,xnew1)
        xnew3 = np.maximum(densities-step_size,xnew2)
        xnew = np.maximum(0.000000001,xnew3)
        
        if (np.sum(xnew)) - volfrac*ele_num > 0:
            l1 = lmid
        else:
            l2 = lmid

        if abs(np.sum(xnew)/(ele_num) -volfrac) < 1e-8:
            # Exit the while loop after it has run at least once
            # print(f"lmid = {lmid}")
            # print(f"Break value: {abs(np.sum(xnew)/(ele_num) -volfrac)}")
            break
    
    
    

    return xnew


def update_density_variables_with_filter_2D(element_nodes, node_coordinates, densities, gradient, radius):
    print("Update density variables with filter is unfinished, remove when finished")

    # get an array of every element center coordinates
    ele_num = element_nodes.shape[0]
    ele_centroids = np.zeros((ele_num,2))

    for i in range(ele_num):
        this_ele_node_coords = node_coordinates[element_nodes[i,:],:]
        #Snippet from ChatGPT for finding centroid
        x, y = this_ele_node_coords[:,0], this_ele_node_coords[:,1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross = x * y1 - x1 * y
        A = 0.5 * cross.sum()
        ele_centroids[i,0] = ((x + x1) * cross).sum() / (6*A)
        ele_centroids[i,1] = ((y + y1) * cross).sum() / (6*A)
        #End centroid finding
    
    # Find which elements are close enough to affect each other


    # update the gradient so that the changes are smoothed 

