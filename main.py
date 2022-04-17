import matplotlib.pyplot as plt
import numpy as np

cpu = np.transpose(np.loadtxt('results.txt'))
gpu = np.transpose(np.loadtxt('results2.txt'))
cpu_plot, gpu_plot = [], []
for i in range(500):
    gpu[0][i] *= 1000
    cpu_plot.append(cpu[0][i])
    gpu_plot.append(gpu[0][i])
xc = np.arange(1, 501)
xg = np.arange(1, 501)
plt.plot(xc, cpu_plot, label='cpu')
plt.plot(xg, gpu_plot, label='gpu')
plt.ylabel('time, msec')
plt.xlabel('matrix size')
# plt.ylim(top=1000)
# plt.xlim(right=100)
plt.grid()
plt.legend()
# plt.savefig('img.jpg')
plt.show()
