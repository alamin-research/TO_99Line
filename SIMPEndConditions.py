import numpy as np

def find_density_max_change(rho, rho_old):
    # Given all of the parameter subject to density, calculate what the total
    # percent change is
    
    # inputs should both be numpy arrays
    
    max_abs_changes = np.max(np.abs(rho - rho_old))
    # print(f"Max change is {max_abs_changes}")
    
    return max_abs_changes