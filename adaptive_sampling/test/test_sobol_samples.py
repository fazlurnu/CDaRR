import matplotlib.pyplot as plt
from adaptive_sampling.utils.sobol_pool import generate_sobol_samples

# Generate samples
samples = generate_sobol_samples(n_samples=512, seed=42)

x = samples[:, 0]
y = samples[:, 1]

# Plot
plt.figure(figsize=(5, 5))
plt.scatter(x, y, s=60)
plt.scatter(x[:32], y[:32], s=60, color = 'tab:red')
plt.xlim(-1.05, 1.05)
plt.ylim(-1.05, 1.05)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sobol Samples (n=16)")
plt.gca().set_aspect("equal", adjustable="box")

plt.show()
