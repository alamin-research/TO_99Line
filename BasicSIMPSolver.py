# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:32:49 2024

@author: caleb
"""

import numpy as np
import matplotlib.pyplot as plt
# The purpose of this file is to solve a given 10x10 Topology Optimization
# problem using the Solid Isotropic Material with Penalization (SIMP) method.
# As this is for a specific data set, it will exclusively be in 2D and will
# expect a mesh size of 10 by 10.

# Based heavily on Sigmund's 99 Line Top Op code:
# https://link.springer.com/article/10.1007/s001580050176

def RectangularSIMP(nelx, nely, volfrac,rmin, fbcx, fbcy, bcf, E, nu, do_graph):
    
    # For this problem the x axis is horizontal with positive right and the y
    # axis is vertical with positive up so that the bottom left element and node
    # can be indexed as 0 and from there read right then up
    
    # all elements are 1 by 1 so a node is located at x = i % (nelx + 1),
    # y = int(i / (nelx + 1))
    
    # nelx = number of elements in the x direction
    # nely = number of elements in the y direction
    # volfrac = the percentage of space that is allowed to be filled
    # rmin = the range of influence that nodes are allowed to have on each other
    # fbc_ = a list of forces applied to the border nodes, should have length of 2*(nelx-1 + nely-1) in _ direction
    # bcf = a list of nodes the same shape as fbc_ where a 1 means the node is fixed
    #       Starting with node 1, it should wrap around the edges
    
    p = np.ones((nely,nelx)) * volfrac  # Initialize the design variable to have "volfrac" density everywhere
    
    change = 1  # Initialize the percent change to be 100% so the loop continues
    loop = 0  # count the number of loops
    
    def unwrapEdge(f,nelx,nely):
        # The purpose of this function is to get the node each wrapped list input refers to
        nodes = []
        for i in range(len(f)):
            if i < nelx:
                # If along the top row, they are the same
                nodes.append(i)
            elif i < (nelx + nely):
                # if along the right side
                nodes.append((i-nelx) * (nelx+1) + nelx)
            elif i < (nelx + nely + nelx):
                # along the bottom
                nodes.append(((nelx+1)*(nely+1))-1-(i-(nelx + nely)))
            else:
                # along the left side
                nodes.append((nely-(i-(nelx + nely + nelx)))*(nelx+1))
        return nodes
    
    wrap_nodes = unwrapEdge(bcf, nelx, nely)
    # print(f"There are {len(wrap_nodes)} nodes around the edges: \n{wrap_nodes}")
    
    # Construct the F matrix with the degrees of freedom
    F = np.zeros(((2*(nely+1)*(nelx+1)),1))
    for i, node in enumerate(wrap_nodes):
        # Add the x forces
        # print(node)
        F[node*2,0] = fbcx[i]
        F[(node*2+1),0] = fbcy[i]
        
    # Construct fixed and free dofs based on bcf
    fixed_dofs = []
    free_dofs = []
    for i,node in enumerate(wrap_nodes):
        if bcf[i] == 1:
            fixed_dofs.append(2*node)
            fixed_dofs.append(2*node + 1)
        else:
            free_dofs.append(2*node)
            free_dofs.append(2*node + 1)
            
    for node in range((nelx+1)*(nely+1)):
        if node not in wrap_nodes:
            free_dofs.append(2*node)
            free_dofs.append(2*node + 1)
            
    #Sort dofs
    free_dofs.sort()
    fixed_dofs.sort()
            
    # Next add all nodes not in
    
    
    #Continue iterating until the change is small
    while change > 0.01:  
        old_p = p  # Store the old density values
        loop += 1
        
        # Get the displacements
        U = FE(nelx, nely, F, free_dofs, fixed_dofs, E, nu, p)
        
        # Calculate the compliance based on 99 Lines 13-24
        k_el = Element_Stiffness_2D(E, nu)
        c = 0 
        dc = np.zeros((nely,nelx))
        for ely in range(nely):
            for elx in range(nelx):
                # select the nodes for this element
                bl, br, tl, tr = getElementNodes((elx+ely*nelx), nelx)
                el_dofs = [2*bl, 2*bl+1, 2*br, 2*br+1, 2*tl, 2*tl+1, 2*tr, 2*tr+1]
                Ue = U[el_dofs,0]
                # print(f"Ue shape is {Ue.shape}")
                
                c = c + p[ely,elx]**3 * np.dot(np.dot(np.transpose(Ue),k_el),Ue)
                
                dc[ely,elx] = -3 * p[ely,elx] ** 2 * np.dot(np.dot(np.transpose(Ue),k_el),Ue)
                
        # dc = filterCompliance(nelx,nely,rmin,p,dc)
        
        p = OC(nelx,nely,p,volfrac,dc)
        
        change = np.max(abs(p-old_p))
        
        if do_graph:
            print(f"Loop {loop} compliance is {c}")
            
        # Put a cap on the number of iterations
        if loop > 50:
            break
        
    if np.sum(p)/(nelx*nely) <1 or np.sum(p)/(nelx*nely) > 0:
        viable = True
    else:
        viable = False
    
    if do_graph:
        # Show final result
        print("Figure out how to make the forces and fixed nodes appear")
        plt.figure(figsize=(8,8))
        plt.imshow(p, cmap='viridis',origin='lower', vmin=0, vmax=1, extent=[0,p.shape[1],0,p.shape[0]])
        plt.colorbar(label='Density')
        plt.title(f"Density for loop {loop}")
        plt.show()
            
        # # Show displacments
        # plot_displacements(U, nelx, nely)
        # # print(f"U is: \n{U}")
            
        print(f"Final volume fraction is {np.sum(p)/(nelx*nely)}")
        
        # ChatGPT to Show forces and loads
        # Generate node coordinates for the unit grid
        x_coords = np.linspace(0, nelx, nelx+1)
        y_coords = np.linspace(0, nely, nely+1)
        X, Y = np.meshgrid(x_coords, y_coords)
    
        # Flatten coordinates to match DOF structure
        node_x_coords = X.flatten()
        node_y_coords = Y.flatten()
        
        # Total number of nodes
        num_nodes = (nelx+1) * (nely+1)
        
        # Extract forces at nodes (reshape F to match DOFs per node)
        Fx = F[0::2].reshape(nely+1, nelx+1)  # x-direction forces
        Fy = F[1::2].reshape(nely+1, nelx+1)  # y-direction forces
        
        # Extract fixed DOFs and their corresponding node indices
        fixed_nodes_x = np.array(fixed_dofs[0::2]) // 2  # Divide by 2 to map DOFs to nodes
        fixed_nodes_y = np.array(fixed_dofs[1::2]) // 2
        
        fixed_x_coords = node_x_coords[fixed_nodes_x]
        fixed_y_coords = node_y_coords[fixed_nodes_y]
    
        # Plot
        plt.figure(figsize=(10, 6))
        plt.quiver(X, Y, Fx, Fy, color='blue', scale=1, scale_units='xy', angles='xy')
        plt.scatter(fixed_x_coords, fixed_y_coords, color='red', label='Fixed Nodes')
        
        # Add grid lines and labels
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Quiver Plot of Forces and Fixed Nodes')
        plt.colorbar(label='Force Magnitude (if scaled)')
        plt.legend()
        plt.axis('equal')
        plt.xlim(-1,nelx+1)
        plt.ylim(-1,nely+2)
        plt.show()
        
    return p, viable

def getElementNodes(element,nelx):
    # Take the element and return the relevant nodes
    bot_left = element + int(element / (nelx + 1))
    bot_right = 1 + bot_left
    top_left = bot_left + nelx + 1
    top_right = top_left +1
    
    return [bot_left, bot_right, top_left, top_right]

def getNodeDOFS(node):
    # take the node and return the relevant degrees of freedom
    return[2*node, 2*node+1]

def FE(nelx, nely, F, freedofs, fixeddofs, E, nu, p):
    k_el = Element_Stiffness_2D(E, nu)
    
    # Initialize the superpositioning variables
    K = np.zeros((2*(nelx+1)*(nely+1),2*(nelx+1)*(nely+1)))
    U = np.zeros((2*(nelx+1)*(nely+1),1))
    
    for x_el in range(nelx):
        for y_el in range(nely):
            # apply the k_el to the dofs for this element
            n = getElementNodes((y_el*(nelx)+x_el), nelx)
            # print(f"For element {(y_el*(nelx)+x_el)} the nodes are {n}")
            dofs = []
            for node in n:
                dofs.extend(getNodeDOFS(node))
                
            # print(f"K shape is {(K[np.ix_(dofs,dofs)]).shape}")
            # print(f"This density shape is {(p[y_el,x_el]**3).shape}")
            # print(f"This element shape is {k_el.shape}")
            K[np.ix_(dofs,dofs)] = K[np.ix_(dofs,dofs)] + p[y_el,x_el]**3 * k_el
            
    # print(f"Shape of K is {(K[np.ix_(freedofs,freedofs)]).shape}")
    # print(f"Shape of F is {(F[freedofs]).shape}")
    
    # print(f"Is K symmetric? {np.allclose(K, np.transpose(K))}")
    
    U[freedofs,:] = np.linalg.solve(K[np.ix_(freedofs,freedofs)], F[freedofs]) 
    
    # print(f"The number of free dofs is {len(freedofs)}")
    # print(f"The number of fixed dofs is {len(fixeddofs)}")
    
    U[fixeddofs,:] = 0
    
    # print(f"U is: \n{U.shape}")
    
    return U
            
def plot_displacements(U, nelx, nely):
    # Total number of nodes
    nnodes_x = nelx + 1
    nnodes_y = nely + 1
    
    # Extract u_x and u_y from U
    ux = U[0::2].reshape(nnodes_y, nnodes_x)  # Horizontal displacements
    uy = U[1::2].reshape(nnodes_y, nnodes_x)  # Vertical displacements
    
    # Create grid for node positions
    x, y = np.meshgrid(range(nnodes_x), range(nnodes_y))
    
    # Create quiver plot
    plt.figure(figsize=(8, 6))
    plt.quiver(x, y, ux, uy, angles='xy', scale_units='xy', scale=1, color='blue')
    #plt.gca().invert_yaxis()  # Invert y-axis for FEM coordinate systems
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Displacement Field (Quiver Plot)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
def Element_Stiffness_2D(E, nu):
    # Taken straight from Sigmund's 99 Lines (87-99)
    k = [1/2 -nu/6, 1/8 + nu/8, -1/4-nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8]
    
    k_el = E/(1 - nu**2) * np.array( [
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ])
    
    # for the above index 0 refers to node 0 (top left) in the x direction for 
    # both row and column, index 1 is the y direction for the same node, etc
    return k_el
    
def filterCompliance(nelx,nely,rmin,x,dc):
    dcn = np.zeros((nely,nelx))
    
    for i in range(nelx):
        for j in range(nely):
            w_sum = 0 
            # max available index changed to 0 to match different indices in python
            for k in range(max(i-round(rmin),0),min(i+round(rmin),nelx)):
                for l in range(max(j-round(rmin),0),min(j+round(rmin),nely)):
                    # print(f"i={i}, j={j}, k={k}, l={l}")
                    fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                    w_sum +=max(0,fac)
                    dcn[j,i] += max(0,fac)*x[l,k]*dc[l,k]
            dcn[j,i] = dcn[j,i]/(x[j,i]*w_sum)
            if (x[j,i]*w_sum) == 0:
                print("There is a 0 in the denominator")
                print(f"X value at j={j}, i={i} is {x[j,i]}")
                print(f"W_sum is {w_sum}")
                input()
            
    return dcn

def OC(nelx,nely,x,volfrac,dc):
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
    

if __name__ == "__main__":
    nelx = 100
    nely = 100
    volfrac = 0.5
    rmin = 1.3
    E = 10
    nu = 0.3
    
    # Pick the nodes at the top corners and bottom corners for bcs
    fbcx = [0]*(2*(nelx+nely))
    fbcy = [0]*(2*(nelx+nely))
    bcf = [0]*(2*(nelx+nely))
    # Apply loads
    # fbcx[nelx+nely] = 1
    fbcx[int(2*nelx+1.5*nely)] = -100
    fbcy[int(0.5*nelx+0*nely)] = -1
    fbcx[int(2*nelx+1.0*nely)] = -1
    fbcy[int(2*nelx+1.0*nely)] = -1
    # fbcy[int(0.5*nelx)+0*nely] = -1
    # Set Fixed nodes
    bcf[int(nelx*1+nely*0)] = 1
    bcf[0] = 1
    
    
    
    print(f"fbcx is {fbcx}")
    print(f"fbcy is {fbcy}")
    print(f"bcf is {bcf}")
    
    
    RectangularSIMP(nelx, nely, volfrac,rmin, fbcx, fbcy, bcf, E, nu, True)