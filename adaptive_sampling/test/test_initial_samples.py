import matplotlib.pyplot as plt
import numpy as np

from adaptive_sampling.sampling.generate_initial_samples import generate_initial_samples
from types import SimpleNamespace

import numpy as np
import pandas as pd

# let's define the bounds here and we generate the samples accordingly

# x1 is resofach, x2 is lookahead time
bounds = SimpleNamespace(
    x1_min=0.9,
    x1_max=1.2,
    x2_min=1.0,
    x2_max=120.0
)

params = SimpleNamespace(
    confidence_interval=15.0,
    confidence_interval_velo=0.5,
    reception_prob=0.95,
    dpsi=45,
)

results = generate_initial_samples(bounds, params, n_samples=32, seed=42)

x1 = np.array([r['x1_resofach'] for r in results])
x2 = np.array([r['x2_lookahead_time'] for r in results])
ipr = np.array([r['sim_results']['overall_ipr'] for r in results])

df = pd.DataFrame({
    'resofach': x1,
    'lookahead_time': x2,
    'ipr': ipr,
})

# Save results
df.to_csv(
    f'results/initial_samples_results_'
    f'{params.confidence_interval}_'
    f'{params.confidence_interval_velo}_'
    f'{params.reception_prob}_'
    f'{params.dpsi}.csv',
    index=False
)

# Masks
unsafe = df['ipr'] < 0.999
safe = ~unsafe

# Plot
plt.figure(figsize=(6, 5))

plt.scatter(
    df.loc[unsafe, 'resofach'],
    df.loc[unsafe, 'lookahead_time'],
    color='tab:red',
    label='Unsafe',
    s=80
)

plt.scatter(
    df.loc[safe, 'resofach'],
    df.loc[safe, 'lookahead_time'],
    color='tab:green',
    label='Safe',
    s=80
)

plt.xlabel("x1 (resofach)")
plt.ylabel("x2 (lookahead time)")
plt.title("Simulation Results")
plt.legend()
plt.show()