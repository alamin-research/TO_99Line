import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm


def plot_2D_boundary_conditions(node_coordinates,fixed_dofs,forces=[],marker_size=2, max_vector_length=1):
        
    # Get the coordinates of everything to be graphed
    x_coords = [point[0] for point in node_coordinates]
    y_coords = [point[1] for point in node_coordinates]
    
    x_fixed_x_coords = []
    x_fixed_y_coords = []

    y_fixed_x_coords = []
    y_fixed_y_coords = []
    
    for dof in fixed_dofs:
        if dof%2 == 0:
            x_fixed_x_coords.append(node_coordinates[int(dof/2)][0])
            x_fixed_y_coords.append(node_coordinates[int(dof/2)][1])
        else:
            y_fixed_x_coords.append(node_coordinates[int(dof/2)][0])
            y_fixed_y_coords.append(node_coordinates[int(dof/2)][1])
            
    # print("\n\nfixed Coords:")
    # print(fixed_dofs)
    # print(x_fixed_x_coords)
    # print(x_fixed_y_coords)
    # print(y_fixed_x_coords)
    # print(y_fixed_y_coords)

    
    scale = max_vector_length / np.max(np.abs(forces))
    for node_id, (x, y) in enumerate(np.array(node_coordinates)):

        Fx = forces[2 * node_id,0]
        Fy = forces[2 * node_id + 1,0]

        # Skip zero vectors (optional)
        if Fx == 0 and Fy == 0:
            continue
        
        # Plot arrow
        plt.arrow(
            x, y,              # start at node position
            Fx * scale, Fy * scale,  # arrow direction & length
            head_width=max_vector_length/5,          # tweak for your mesh size
            head_length=max_vector_length/3,
            fc='green',
            ec='green'
        )
    
            
    # Plot the points
    plt.plot(x_coords, y_coords, 'k.', markersize=1)  
    plt.plot(x_fixed_x_coords,x_fixed_y_coords, 'g+', markersize=marker_size)
    plt.plot(y_fixed_x_coords,y_fixed_y_coords, 'gx', markersize=marker_size)

    # Optionally, label the axes and show the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Points")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(2)

def plot_2D_weight_gradient(element_nodes, node_coordinates, fixed_dofs, gradient_in, iteration_num=-1, forces=[], marker_size=2):

    if forces !=[]:
        print("Display of forces not implemented yet")

    node_coordinates = np.array(node_coordinates)
    element_nodes = np.array(element_nodes)
    gradient = np.asarray(gradient_in).reshape(-1)

    # Take the weight gradient and plot it so that it is clear what the effect of the previous iteration is
    # a blue result should mean that the element shown in increasing and red means decreasing

    polys = []
    for elem in element_nodes:
        poly = node_coordinates[elem]   # (4×2) coordinates for the element
        polys.append(poly)

    fig, ax = plt.subplots()
    
    # PolyCollection (filled quads colored by gradient)
    # Create a diverging normalization centered at zero
    norm = TwoSlopeNorm(vcenter=0.0)

    coll = PolyCollection(
        polys,
        array=gradient,
        cmap='bwr',      # blue ↔ white ↔ red colormap
        norm=norm,       # forces white at 0
        edgecolor='k',
        linewidth=0.2
    )

    ax.add_collection(coll)

    # Get the coordinates of everything to be graphed
    x_coords = [point[0] for point in node_coordinates]
    y_coords = [point[1] for point in node_coordinates]
    
    x_fixed_x_coords = []
    x_fixed_y_coords = []

    y_fixed_x_coords = []
    y_fixed_y_coords = []
    
    for dof in fixed_dofs:
        if dof%2 == 0:
            x_fixed_x_coords.append(node_coordinates[int(dof/2)][0])
            x_fixed_y_coords.append(node_coordinates[int(dof/2)][1])
        else:
            y_fixed_x_coords.append(node_coordinates[int(dof/2)][0])
            y_fixed_y_coords.append(node_coordinates[int(dof/2)][1])
            
    # Plot the points
    plt.plot(x_coords, y_coords, 'k.', markersize=1)  
    plt.plot(x_fixed_x_coords,x_fixed_y_coords, 'g+', markersize=marker_size)
    plt.plot(y_fixed_x_coords,y_fixed_y_coords, 'gx', markersize=marker_size)

    # Optionally, label the axes and show the plot
    ax.set_aspect('equal')
    plt.colorbar(coll, ax=ax, label='Gradient Value', norm=norm)
    plt.xlabel("x")
    plt.ylabel("y")
    if iteration_num == -1:
        plt.title("Element-wise Gradient Field (blue=positive, red=negative)")
    else:
        plt.title(f"Element-wise Gradient Field {iteration_num} (blue=positive, red=negative)")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(2)