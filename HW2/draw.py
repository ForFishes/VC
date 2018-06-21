import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import numpy as np

Frames = np.arange(1,300)
print(Frames)
with open(sys.argv[1], 'r') as f:
    a = f.read().split('\n')
FULL = np.array(' '.join(a[1:11]).split(' ')[1:], dtype=int)
TSS = np.array(' '.join(a[12:22]).split(' ')[1:], dtype=int)
NTSS = np.array(' '.join(a[23:33]).split(' ')[1:], dtype=int)
_2D_Log = np.array(' '.join(a[34:44]).split(' ')[1:], dtype=int)
Orthogonal = np.array(' '.join(a[45:55]).split(' ')[1:], dtype=int)

plt.figure()
plt.plot(Frames, FULL, label='Full')
plt.plot(Frames, TSS, label='TSS')
plt.plot(Frames, NTSS, label='NTSS')
plt.plot(Frames, _2D_Log, label='2D Log')
plt.plot(Frames, Orthogonal, label='Orthogonal')
plt.legend(fontsize='xx-large')
plt.title("Foreman")
plt.show()
