import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# Change the backend to TkAgg for interactive plotting
plt.switch_backend('TkAgg')

refSig = 3
plasmaSig = 2.5

errorPercent = np.arange(0, 100, 1)

trueSig = np.log(refSig/plasmaSig)

posErrorSig = np.log(refSig/(plasmaSig*(1 + errorPercent/100)))
negErrorSig = np.log(refSig/(plasmaSig*(1 - errorPercent/100)))

posPercentError = 100 * np.abs((trueSig - posErrorSig) / trueSig)
negPercentError = 100 * np.abs((trueSig - negErrorSig) / trueSig)

fig = plt.figure(figsize=(10, 6), tight_layout=True)
ax = fig.add_subplot(111)

ax.plot(errorPercent, posPercentError, label='Positive Error', linewidth=2)
ax.plot(errorPercent, negPercentError, label='Negative Error', linewidth=2)

ax.set_xlim(0, 20)
ax.set_ylim(0, 100)
ax.legend()

ax.set_xlabel('Raw Signal Error (%)')
ax.set_ylabel(r'$\int n_p \cdot dl$ Error (%)')

plt.show()