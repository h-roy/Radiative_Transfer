import sobol_seq
import numpy as np

def assemble_dataset(tau_max, nu_max, n_collocation, n_boundary, seed):
    # Sample interior points using sobol sequences and boundary points
    # Interior Points
    skip = seed
    interior_data = np.full((n_collocation, 3), np.nan)
    for j in range(n_collocation):
        seed = j + skip
        interior_data[j, :], next_seed = sobol_seq.i4_sobol(3, seed)


    #First column of data is /nu, Second Column is /tau and Third Column is /mu

    interior_data[:, 0] = interior_data[:, 0] * 2 * nu_max - nu_max
    interior_data[:, 1] = interior_data[:, 1] * tau_max
    #make sure no zeros here
    interior_data[:, 2] = interior_data[:, 2] * 2 - 1

    #Sample Boundary Points ( tau = 0 and tau_max)
    n_b_1 = int(n_boundary/2)
    n_b_2 = n_boundary - n_b_1
    boundary_data_1 = np.full((n_b_1, 3), np.nan)
    for j in range(n_b_1):
        seed = j + skip
        boundary_data_1[j, :], next_seed = sobol_seq.i4_sobol(3, seed)


    #First column of data is /nu, Second Column is /tau and Third Column is /mu

    boundary_data_1[:, 0] = boundary_data_1[:, 0] * 2 * nu_max - nu_max
    boundary_data_1[:, 1] = tau_max
    #make sure no zeros here
    boundary_data_1[:, 2] = boundary_data_1[:, 2] * 2 - 1

    boundary_data_2 = np.full((n_b_2, 3), np.nan)
    for j in range(n_b_2):
        seed = j + skip
        boundary_data_2[j, :], next_seed = sobol_seq.i4_sobol(3, seed)


    #First column of data is /nu, Second Column is /tau and Third Column is /mu

    boundary_data_2[:, 0] = boundary_data_2[:, 0] * 2 * nu_max - nu_max
    boundary_data_2[:, 1] = 0
    #make sure no zeros here
    boundary_data_2[:, 2] = boundary_data_2[:, 2] * 2 - 1

    return interior_data, boundary_data_1, boundary_data_2
