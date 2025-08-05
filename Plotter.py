import numpy as np
import matplotlib.pyplot as plt


def plot_2D_boundary_conditions(node_coordinates,fixed_dofs,forces=[],marker_size=2):
    
    if forces !=[]:
        print("Display of forces not implemented yet")
        
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
            
    print("\n\nfixed Coords:")
    print(fixed_dofs)
    print(x_fixed_x_coords)
    print(x_fixed_y_coords)
    print(y_fixed_x_coords)
    print(y_fixed_y_coords)
            
    # Plot the points
    plt.plot(x_coords, y_coords, 'k.', markersize=1)  
    plt.plot(x_fixed_x_coords,x_fixed_y_coords, 'r+', markersize=marker_size)
    plt.plot(y_fixed_x_coords,y_fixed_y_coords, 'rx', markersize=marker_size)

    # Optionally, label the axes and show the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Points")
    plt.grid(True)
    plt.show()