import segyio
import numpy as np
import matplotlib.pyplot as plt

segy_path = "seam.sgy"  #file name

with segyio.open(segy_path, "r", ignore_geometry=True) as f:
    f.mmap()

    n_traces = f.tracecount             # all traces
    n_samples = len(f.samples)          # sample count in all traces

    print("sum trace:", n_traces)
    print("sample count (her trace):", n_samples)

    # read all traces
    seismic = np.zeros((n_traces, n_samples))
    for i in range(n_traces):
        seismic[i, :] = f.trace[i]

print("data size:", seismic.shape)

# visualization
plt.figure(figsize=(12, 6))
plt.imshow(seismic[:100].T, aspect='auto', cmap='seismic')
plt.title("SEAM data - first 100 Trace")
plt.xlabel("Trace No")
plt.ylabel("Time Series")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()
patch_size = 64  # Patch size 64x64
stride = 64      # Slide every step

patches = []     # Sum all patches

# Height and width
height, width = seismic.shape


for i in range(0, height - patch_size + 1, stride):
    for j in range(0, width - patch_size + 1, stride):
        patch = seismic[i:i+patch_size, j:j+patch_size]
        patches.append(patch)

print(f"Sum {len(patches)} patch extracted")

# NumPy
patches = np.array(patches)

# Visualize 4 patch
plt.figure(figsize=(8, 8))
for k in range(4):
    plt.subplot(2, 2, k + 1)
    plt.imshow(patches[k], cmap='seismic', aspect='auto')
    plt.title(f"Patch {k+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
