import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
class MagnetiteModel:
    def __init__(self, nx, ny, nz=1):
        """
        Initialize magnetite model with specific parameters.
        
        Parameters:
        -----------
        nx, ny, nz : int
            Dimensions of the lattice
        """
        # Lattice parameters
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # Magnetite-specific parameters (in eV)
        self.t_dd = 0.5    # Fe-Fe hopping
        self.t_pd = 2.0    # Fe-O hopping
        self.U_fe = 4.0    # On-site Coulomb repulsion for Fe
        self.JH = 0.9      # Hund's coupling
        self.delta_cf = 2.0 # Crystal field splitting
        
        # Fe3O4 specific orbital energies
        self.E_fe_t2g = 0.0    # Reference energy
        self.E_fe_eg = self.E_fe_t2g + self.delta_cf
        self.E_oxygen = -3.0   # Oxygen 2p orbital energy
        
    def create_hamiltonian(self, include_edges=True):
        """
        Creates the Hamiltonian matrix with edge states.
        
        Parameters:
        -----------
        include_edges : bool
            If True, uses open boundary conditions to study edge states
        """
        n_sites = self.nx * self.ny * self.nz
        n_orb_fe = 5  # 3d orbitals
        n_orb_o = 3   # 2p orbitals
        
        # Total dimension includes Fe 3d and O 2p orbitals
        dim = (n_sites * n_orb_fe + n_sites * n_orb_o) * 2  # *2 for spin
        
        row, col, data = [], [], []
        
        # Implement hopping terms
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    site = self.get_site_index(ix, iy, iz)
                    
                    # Fe-Fe hopping
                    neighbors = self.get_neighbors(ix, iy, iz, include_edges)
                    for neighbor in neighbors:
                        self._add_fe_fe_hopping(site, neighbor, row, col, data)
                    
                    # Fe-O hopping
                    self._add_fe_o_hopping(site, row, col, data)
                    
                    # On-site terms
                    self._add_onsite_terms(site, row, col, data)
        
        H = sparse.csr_matrix((data, (row, col)), shape=(dim, dim))
        return (H + H.conjugate().transpose()) * 0.5
    
    def _add_fe_fe_hopping(self, site1, site2, row, col, data):
        """Add Fe-Fe hopping terms."""
        n_orb = 5
        for orb in range(n_orb):
            for spin in [0, 1]:
                idx1 = self.get_state_index(site1, orb, spin, is_fe=True)
                idx2 = self.get_state_index(site2, orb, spin, is_fe=True)
                
                row.append(idx1)
                col.append(idx2)
                data.append(-self.t_dd)
    
    def _add_fe_o_hopping(self, site, row, col, data):
        """Add Fe-O hopping terms."""
        n_orb_fe = 5
        n_orb_o = 3
        
        for orb_fe in range(n_orb_fe):
            for orb_o in range(n_orb_o):
                for spin in [0, 1]:
                    idx_fe = self.get_state_index(site, orb_fe, spin, is_fe=True)
                    idx_o = self.get_state_index(site, orb_o, spin, is_fe=False)
                    
                    row.append(idx_fe)
                    col.append(idx_o)
                    data.append(-self.t_pd)
    
    def _add_onsite_terms(self, site, row, col, data):
        """Add on-site energy terms and interactions."""
        # Fe 3d orbital energies
        for orb in range(5):
            for spin in [0, 1]:
                idx = self.get_state_index(site, orb, spin, is_fe=True)
                energy = self.E_fe_t2g if orb < 3 else self.E_fe_eg
                
                row.append(idx)
                col.append(idx)
                data.append(energy)
        
        # O 2p orbital energies
        for orb in range(3):
            for spin in [0, 1]:
                idx = self.get_state_index(site, orb, spin, is_fe=False)
                
                row.append(idx)
                col.append(idx)
                data.append(self.E_oxygen)
    
    def analyze_edge_states(self):
        """Analyze edge states by comparing bulk and edge eigenvalues."""
        # Create Hamiltonians with and without edges
        H_bulk = self.create_hamiltonian(include_edges=False)
        H_edge = self.create_hamiltonian(include_edges=True)
        
        # Calculate eigenvalues using sparse.linalg.eigsh
        n_states = 20  # Number of states to analyze
        eigenvals_bulk = linalg.eigsh(H_bulk, k=n_states, which='SA', return_eigenvectors=False)
        eigenvals_edge = linalg.eigsh(H_edge, k=n_states, which='SA', return_eigenvectors=False)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_states), eigenvals_bulk, 'bo-', label='Bulk states')
        plt.plot(range(n_states), eigenvals_edge, 'ro-', label='With edges')
        plt.xlabel('State index')
        plt.ylabel('Energy (eV)')
        plt.title('Edge States Analysis in Magnetite')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return eigenvals_bulk, eigenvals_edge
    
    def get_site_index(self, ix, iy, iz):
        """Convert 3D coordinates to site index."""
        return ix + self.nx * (iy + self.ny * iz)
    
    def get_neighbors(self, ix, iy, iz, include_edges):
        """Get indices of neighboring sites."""
        neighbors = []
        
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = ix + dx, iy + dy, iz + dz
            
            if include_edges:
                # Open boundary conditions
                if 0 <= nx < self.nx and 0 <= ny < self.ny and 0 <= nz < self.nz:
                    neighbors.append(self.get_site_index(nx, ny, nz))
            else:
                # Periodic boundary conditions
                nx = nx % self.nx
                ny = ny % self.ny
                nz = nz % self.nz
                neighbors.append(self.get_site_index(nx, ny, nz))
        
        return neighbors
    
    def get_state_index(self, site, orbital, spin, is_fe):
        """Get index in the Hilbert space."""
        n_orb_fe = 5
        n_orb_o = 3
        n_sites = self.nx * self.ny * self.nz
        
        if is_fe:
            return site * (n_orb_fe * 2) + orbital * 2 + spin
        else:
            offset = n_sites * n_orb_fe * 2
            return offset + site * (n_orb_o * 2) + orbital * 2 + spin

# Example usage
def main():
    # Create a 4x4x1 magnetite lattice
    model = MagnetiteModel(nx=4, ny=4, nz=1)
    
    # Analyze edge states
    eigenvals_bulk, eigenvals_edge = model.analyze_edge_states()
    
    # Print energy differences between edge and bulk states
    print("\nEnergy differences between edge and bulk states:")
    for i in range(len(eigenvals_bulk)):
        diff = eigenvals_edge[i] - eigenvals_bulk[i]
        if abs(diff) > 0.1:  # Significant difference indicating possible edge state
            print(f"State {i}: {diff:.3f} eV")

if __name__ == "__main__":
    main()