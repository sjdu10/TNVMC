import matplotlib.pyplot as plt
import numpy as np
import h5py 
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8*3)})
#plt.rcParams.update({'font.family':'serif'})
#plt.rcParams.update({'text.usetex':True})

fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1)
f = h5py.File('energy.hdf5','r')
data = f['data'][:]
f.close()
dE = np.array([data[ix+1]-data[ix] for ix in range(len(data)-1)])
ax1.plot(np.arange(len(data)),data, linestyle='', marker='o', label=r'TNS ($D=6,\chi = 12$)')
ax1.plot(np.arange(len(data)), [-0.498886]*len(data), 'g--', label='NNS (-0.498886)')
ax2.plot(np.arange(1,len(data)),np.fabs(dE), linestyle='', marker='o')
ax4.plot(np.arange(len(data)), (np.array(data)-np.array([-0.498886]*len(data)))/(0.498886), 'g', label=r'$(E_{TNS}-E_{NNS})/E_{NNS}$')
ax5.plot(np.arange(len(data)), np.array(data)-np.array([-0.498886]*len(data)), 'r', label=r'$E_{TNS}-E_{NNS}$')

#f = h5py.File('gradient.hdf5','r')
#data = f['data'][:]
#f.close()
#ax3.plot(np.arange(start,start+len(data)),data, linestyle='', marker='o', color=color,label=label)

f = h5py.File('err.hdf5','r')
data = f['data'][:]
f.close()
ax3.plot(np.arange(len(data)),data, linestyle='', marker='o')

ax5.set_xlabel('step')
ax1.set_ylabel('E')
ax2.set_ylabel('dE')
ax3.set_ylabel('statistical error')
ax2.set_yscale('log')
ax3.set_yscale('log')
ax4.set_ylabel('Relative E difference from NNS')
ax4.set_yscale('log')
ax5.set_ylabel(r'$\Delta E$')
ax5.legend()
ax4.legend()
#ax2.set_ylim((.0001,.1))
#ax3.set_ylim((.01,1.))
#ax4.set_ylim((.0005,.02))
plt.subplots_adjust(left=0.2, bottom=0.05, right=0.99, top=0.99)
ax1.legend()
fig.savefig("VMC_data.png", dpi=250)