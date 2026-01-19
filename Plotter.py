import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm

# Keep these global so the figure is reused
_fig = None
_ax = None
_collection = None
_colorbar = None


def plot_2D_boundary_conditions(node_coordinates,fixed_dofs,forces=[],marker_size=2, max_vector_length=1, title=-1):
        
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
    plt.plot(x_coords, y_coords, 'k.', markersize=marker_size)  
    plt.plot(x_fixed_x_coords,x_fixed_y_coords, 'g+', markersize=marker_size)
    plt.plot(y_fixed_x_coords,y_fixed_y_coords, 'gx', markersize=marker_size)

    # Optionally, label the axes and show the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Points")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(2)

def plot_2D_displacments(node_coordinates_in, displacements_in,exaggeration=1):

    # abs_disp = np.abs(displacements_in) * exaggeration

    # if np.any(abs_disp != 0):
    #     min_value = np.min(abs_disp[abs_disp != 0])
    #     max_value = np.max(abs_disp[abs_disp != 0])
    #     print(f"After exaggeration, the min change is {min_value }")
    #     print(f"After exaggeration, the min change is {max_value}")
    # else:
    #     print("displacements all 0")

    # find the displaced coordinates
    node_coordinates = np.array(node_coordinates_in)
    displacemented_nodes = exaggeration * np.array(displacements_in).reshape(-1,2) + node_coordinates

    # Get the coordinates of everything to be graphed
    x_coords = [point[0] for point in node_coordinates]
    y_coords = [point[1] for point in node_coordinates]
    dx_coords = [point[0] for point in displacemented_nodes]
    dy_coords = [point[1] for point in displacemented_nodes]
    
    # Plot the points
    plt.plot(x_coords, y_coords, 'k.', markersize=6)      
    plt.plot(dx_coords, dy_coords, 'r.', markersize=3)  
    # Optionally, label the axes and show the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot of displacements")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(2)

def plot_2D_mesh_densities(element_nodes, node_coordinates, densities, iteration_count):

    global _fig, _ax, _collection, _colorbar

    # Build polygon list once per call (cheap)
    polys = []
    for elem in element_nodes:
        polys.append(node_coordinates[elem])

    polys = np.asarray(polys)

    # First call: initialize plot
    if _fig is None:
        plt.ion()   # interactive mode ON

        _fig, _ax = plt.subplots()

        _collection = PolyCollection(
            polys,
            array=densities,
            cmap="gray_r",        # <-- reversed grayscale (0=white,1=black)
            edgecolors="none",    # faster rendering
            linewidths=0.0
        )

        # Force color limits
        _collection.set_clim(0.0, 1.0)

        _ax.add_collection(_collection)

        _ax.autoscale()
        _ax.set_aspect("equal")

        _colorbar = plt.colorbar(_collection, ax=_ax)
        _colorbar.set_label("Density")

        _ax.set_title(f"Topology Density Field. Iteration {iteration_count}")
        _ax.set_xlabel("X")
        _ax.set_ylabel("Y")

        plt.show(block=False)

    # Subsequent calls: update only density values
    else:
        _collection.set_array(densities)

    # Refresh canvas without blocking
    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()
    plt.pause(0.001)



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


def plot_global_stiffness_sensitivities(K_global, node_coordinates, scale=1.0):
    """
    Plots four graphs visualizing stiffness sensitivities between nodes.

    Parameters
    ----------
    K_global : scipy.sparse.csr_matrix
        Global stiffness matrix with interleaved DOFs.
    node_coordinates : ndarray (n_nodes, 2)
        Node coordinates [x, y].
    scale : float
        Scaling factor for line thickness.
    """

    n_nodes = node_coordinates.shape[0]

    # Convert once for fast access
    K = K_global.tocsr()
    scale = scale / np.max(np.abs(K_global.data)) # Prevent the lines from being huge

    plots = [
        ("ux → ux", 0, 0, "red"),
        ("uy → uy", 1, 1, "blue"),
        ("ux → uy", 0, 1, "green"),
        ("uy → ux", 1, 0, "purple"),
    ]

    for title, row_dof_offset, col_dof_offset, color in plots:
        plt.figure(figsize=(7, 7))

        # Plot all nodes
        plt.scatter(
            node_coordinates[:, 0],
            node_coordinates[:, 1],
            c="black",
            s=10,
            zorder=3,
        )

        for i in range(n_nodes):
            row_dof = 2 * i + row_dof_offset

            row_start = K.indptr[row_dof]
            row_end = K.indptr[row_dof + 1]

            cols = K.indices[row_start:row_end]
            vals = K.data[row_start:row_end]

            for col_dof, val in zip(cols, vals):
                # Only look at requested column DOF type
                if col_dof % 2 != col_dof_offset:
                    continue

                j = col_dof // 2

                if i == j:
                    continue

                x = [node_coordinates[i, 0], node_coordinates[j, 0]]
                y = [node_coordinates[i, 1], node_coordinates[j, 1]]

                plt.plot(
                    x,
                    y,
                    color=color,
                    linewidth=scale * abs(val),
                    alpha=0.6,
                    zorder=1,
                )

        plt.title(title)
        plt.axis("equal")
        plt.grid(False)
        plt.show()
