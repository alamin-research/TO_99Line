import numpy as np

"""
This file is meant to be a library of functions to get the nodal locations
and other boundary conditions for the problem initial state

Every function here should return a numpy array called element_nodes and a
numpy array called node_coordinates. These are chosen to eventually align
with Abaqus .inp files

element_nodes should be a numpy array where each each row relates to an element
and each spot in that row contains the nodes numbers of that define that element

node_coordinates should be a numpy array where each row relates to a node and
the row its self is the coordinate location of the node in x,y,z format
"""
## Get Nodal positions

def generate_2D_unit_cell_node_grid(nelx, nely):
    # Generate the element_nodes array and node_coordinates from scratch based
    # on the number of elements given as inputs
    
    # Go through each element and assign nodes to that element. The 0th element
    # will be will have the 0th node at its bottom left corner (bl). Node 1 will
    # be the bottom right (br) and the top left (tl) and right (tr) will be 
    # nodes nelx+1 and nelx+2 respectively.
    element_nodes_list = []
    
    # Checklist
    checklist = []
    
    for y in range(nely):
        for x in range(nelx):
            bl = x + y * (nelx+1)
            br = x + y * (nelx+1) + 1
            tl = x + (y+1) * (nelx+1)
            tr = x + (y+1) * (nelx+1) + 1
            
            if bl not in checklist:
                checklist.append(bl)
            if br not in checklist:
                checklist.append(br)
            if tl not in checklist:
                checklist.append(tl)
            if tr not in checklist:
                checklist.append(tr)
            
            element_nodes_list.append([bl,br,tr,tl]) # Force CCW listing
                
    for i in range((nelx+1)*(nely+1)):
        if i not in checklist:
            print(f"i={i} is not in checklist")
            input()
            
    # Go through each node and assign its position based on a side length of 1
    node_coordinates = []
    for y in range(nely+1):
        for x in range(nelx+1):
            node_coordinates.append([x, y])
            
            
    return np.array(element_nodes_list), np.array(node_coordinates)


## Get Fixed Node indices

def add_rolling_edge_fixed_in_y_to_2D_rectangular_mesh(current_fixed, edge_x_position, node_coordinates):
    dofs_to_add = [2*i + 1 for i, coords in enumerate(node_coordinates) if coords[0] == edge_x_position]  # Fix the edge where x=0 in the y direction for symmetry
    current_fixed.extend(dofs_to_add)
    return list(set(current_fixed)) # Switch to a set to remove redundancies

def add_fixed_dof_to_node_near_position_to_2D_rectangular_mesh(current_fixed, node_x_position, node_y_position, radius, node_coordinates, fix_in_x=True, fix_in_y=True):
    
    nodes_to_add = []
    for i, coords in enumerate(node_coordinates):
        if np.sqrt((node_x_position-coords[0])**2 + (node_y_position-coords[1])**2) <= radius:
            nodes_to_add.append(i)
            
    dofs_to_add = []
    for node in nodes_to_add:
        if fix_in_x:
            dofs_to_add.append(2*node)
        if fix_in_y:
            dofs_to_add.append(2*node+1)
            
    current_fixed.extend(dofs_to_add)
    return list(set(current_fixed)) # Switch to a set to remove redundancies
            
            
if __name__ == "__main__":
    
    # Testing methods
    element_nodes, node_coordinates = generate_2D_unit_cell_node_grid(100, 50)
    
