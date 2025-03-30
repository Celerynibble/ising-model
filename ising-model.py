import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(L):
    return np.random.choice([-1, 1], size=(L, L))

def compute_energy(lattice, J=1, H=0):
    energy = 0
    L = lattice.shape[0]
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbors = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            energy += -J * S * neighbors
    energy -= H * np.sum(lattice)  #contribution of the external field
    return energy/2  #counting each bond twice

def metropolis_step(lattice, beta, J=1, H=0):
    L = lattice.shape[0]
    for n in range(L*L):
        i, j = np.random.randint(0, L, size=2)
        S = lattice[i, j]
        neighbors = lattice[(i+1)%L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
        dE = 2 * J * S * neighbors + 2 * H * S  #including the magnetic field term
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1

def simulate(L, T, steps, equilibration_steps, H=0):
    beta = 1 / T
    lattice = initialize_lattice(L)
    initial_lattice = np.copy(lattice)
    energies = []
    magnetizations = []
    
    for n in range(equilibration_steps):
        metropolis_step(lattice, beta, H=H)

    for n in range(steps):
        metropolis_step(lattice, beta, H=H)
        energy = compute_energy(lattice, H=H)
        magnetization = np.sum(lattice) / (L * L)
        energies.append(energy)
        magnetizations.append(magnetization)
    
    #thermodynamic properties
    E_mean = np.mean(energies)
    M_mean = np.mean(magnetizations)
    C = beta**2 * np.var(energies)
    X = beta*np.var(magnetizations)*L**2
    
    return E_mean/(L*L), M_mean, C, X, initial_lattice, lattice


L = 20  #lattice size
T = 2.5  #temperature
steps = 5000
equi_steps = 2000
H = 0.1  #value of the external field

E, M, C, X, initial_lattice, final_lattice = simulate(L, T, steps, equi_steps, H)


print(f"Energy per site: {E:.3f}")
print(f"Magnetization per site: {M:.3f}")
print(f"Specific heat: {C:.3f}")
print(f"Susceptibility: {X:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(initial_lattice, cmap='gray')
ax[0].set_title("Initial Lattice Configuration")
ax[1].imshow(final_lattice, cmap='gray')
ax[1].set_title("Final Lattice Configuration")
plt.show()
