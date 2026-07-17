import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

class Lattice:
    def __init__(self, basis):
        """
        Intitalizes a lattices define by a basis matrix

        Parameters:
        basis: An [n x n] matrix that consists of basis vectors as its ROWS
        """

        self.basis = np.array(basis, dtype=float)
        self.dimension = self.basis.shape[0]

        if self.basis.shape[0] != self.basis.shape[1]:
            raise ValueError("The basis matrix must be full rank!")

    def point(self, coeff):
        """
        Computes a point of the lattice using integer coefficients
        """

        coeff = np.array(coeff, dtype=int)

        # Calculates the linear combination of the rows using matrix multiplication
        return coeff @ self.basis

    def generate_points(self, coord_range):
        """
        Generates all the lattice points within a grid bounded by the integer coefficients
        """

        if self.dimension != 2:
            raise NotImplementedError("Lattices can only generate in 2D (currently)")

        r = np.arange(coord_range[0], coord_range[1] + 1)

        c1, c2 = np.meshgrid(r,r)
        c_flat = np.column_stack((c1.ravel(), c2.ravel()))

        points = c_flat @ self.basis
        return points
    
    def determinant(self):
        """
        Computes the determinant of the lattice
        """
        return np.abs(np.linalg.det(self.basis))

    def plot_lattice(self, coord_range=(-5, 5), filename="lattice_output.png"):
        """
        Plots out both the lattice points and basis vector
        coord_range: (n x n) tuple of the coordinatre range for the points
        """

        if self.dimension != 2:
            raise NotImplementedError("Plotting only support 2D lattices")

        points = self.generate_points(coord_range)


        plt.figure(figsize=(6,8))

        plt.scatter(points[:, 0], points[:, 1], color="blue", s=20, zorder=3, label="Lattice Points $\mathcal{L}(B)$")

        plt.quiver(0, 0, self.basis[0, 0], self.basis[0, 1], angles='xy', scale_units='xy', 
                   scale=1, color='#e74c3c', width=0.007, zorder=5, label='$b_1$')
        plt.quiver(0, 0, self.basis[1, 0], self.basis[1, 1], angles='xy', scale_units='xy', 
                   scale=1, color='#3498db', width=0.007, zorder=5, label='$b_2$')
        
        # Decorate the plot
        plt.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        plt.axvline(0, color='black', linewidth=0.8, alpha=0.5)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axis('equal')
        
        # Set dynamic limits based on the generated points to keep everything framed beautifully
        margin = 1.2
        plt.xlim(coord_range[0] * np.max(np.abs(self.basis)) * margin, coord_range[1] * np.max(np.abs(self.basis)) * margin)
        plt.ylim(coord_range[0] * np.max(np.abs(self.basis)) * margin, coord_range[1] * np.max(np.abs(self.basis)) * margin)
        
        plt.title(u"2D Lattice System\n$\det(\mathcal{L}) = " + f"{self.determinant():.2f}$", fontsize=12)
        plt.legend(loc='upper right')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(u"Successfully generated lattice plot: '{filename}'")

if __name__ == "__main__":

    bad_basis = [[2.0, 0.0], 
             [1.5, 0.5]]

    # Create your object
    L = Lattice(bad_basis)

        # Print the volume of the lattice unit cell
    print(f"Lattice Determinant: {L.determinant()}")

    # Generate a single targeted point: 3*b1 + (-2)*b2
    print(f"Lattice point for coefficients [3, -2]: {L.point([3, -2])}")

    # Save the visualization grid out to disk
    L.plot_lattice(coord_range=(-6, 6), filename="skewed_lattice.png")



    


