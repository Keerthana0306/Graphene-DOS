import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
def create_magnetite_hamiltonian(n_sites, t_hop=1.0, U=4.0, JH=0.9):
    """
    Creates the Hamiltonian matrix for a magnetite (Fe3O4) crystal lattice.
    
    Parameters:
    -----------
    n_sites : int
        Number of iron sites in the unit cell
    t_hop : float
        Hopping parameter between Fe sites
    U : float
        On-site Coulomb repulsion
    JH : float
        Hund's coupling constant
    
    Returns:
    --------
    sparse.csr_matrix
        The Hamiltonian matrix in sparse format
    """
    # Number of orbitals per Fe site (3d orbitals)
    n_orb = 5
    
    # Total Hilbert space dimension
    dim = n_sites * n_orb * 2  # *2 for spin
    
    # Initialize lists for sparse matrix construction
    row = []
    col = []
    data = []
    
    # Kinetic term (hopping between sites)
    for i in range(n_sites):
        for orb in range(n_orb):
            for spin in [0, 1]:
                # Nearest neighbor hopping
                neighbors = get_neighbors(i, n_sites)
                for j in neighbors:
                    idx1 = get_state_index(i, orb, spin, n_orb)
                    idx2 = get_state_index(j, orb, spin, n_orb)
                    
                    row.append(idx1)
                    col.append(idx2)
                    data.append(-t_hop)
    
    # On-site interaction terms
    for i in range(n_sites):
        for orb1 in range(n_orb):
            for orb2 in range(n_orb):
                # Intra-orbital Coulomb repulsion
                if orb1 == orb2:
                    idx_up = get_state_index(i, orb1, 0, n_orb)
                    idx_dn = get_state_index(i, orb1, 1, n_orb)
                    
                    row.append(idx_up)
                    col.append(idx_up)
                    data.append(U * 0.5)
                    
                    row.append(idx_dn)
                    col.append(idx_dn)
                    data.append(U * 0.5)
                
                # Inter-orbital Hund's coupling
                else:
                    for spin1 in [0, 1]:
                        for spin2 in [0, 1]:
                            idx1 = get_state_index(i, orb1, spin1, n_orb)
                            idx2 = get_state_index(i, orb2, spin2, n_orb)
                            
                            # Parallel spins
                            if spin1 == spin2:
                                row.append(idx1)
                                col.append(idx2)
                                data.append(-JH)
                            
                            # Anti-parallel spins
                            else:
                                row.append(idx1)
                                col.append(idx2)
                                data.append(JH)
    
    # Create sparse matrix
    H = sparse.csr_matrix((data, (row, col)), shape=(dim, dim))
    
    # Make Hermitian
    H = (H + H.conjugate().transpose()) * 0.5
    
    return H

def get_state_index(site, orbital, spin, n_orb):
    """Helper function to get the index in the Hilbert space."""
    return site * (n_orb * 2) + orbital * 2 + spin

def get_neighbors(site, n_sites):
    """
    Helper function to get nearest neighbors in the lattice.
    This is a simplified version - modify based on actual crystal structure.
    """
    neighbors = [(site + 1) % n_sites, (site - 1) % n_sites]
    return neighbors

# Example usage:
def main():
    # Create Hamiltonian for a small system
    n_sites = 4  # 4 iron sites
    H = create_magnetite_hamiltonian(n_sites)
    
    # Calculate eigenvalues
    eigenvalues = linalg.eigsh(H, k=6, which='SA', return_eigenvectors=False)
    print("Lowest 6 energy eigenvalues:")
    print(eigenvalues)

if __name__ == "__main__":
    main()