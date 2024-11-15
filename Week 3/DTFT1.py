import numpy as np
import matplotlib.pyplot as plt
def dtft(x):
	y=[]
	w=np.arange(-np.pi,np.pi,0.001*np.pi)
	for f in w:
		s=0
		for n in range(0,len(x)):
			s=s+x[n]*np.exp(-1j*f*n)
		y.append(s)
	magnitude=np.abs(y)
	phase=np.angle(y)
	return w,magnitude,phase

n=np.arange(0,500)
x=np.sin(2*np.pi*(200/8000)*n)
f,m,p=dtft(x)
plt.subplot(3,1,1)
plt.plot(x)
plt.title("Original signal")
plt.xlabel("frequency")
plt.ylabel("Amplitude")

plt.subplot(3,1,2)
plt.plot(f,m)
plt.title("magnitude spectrum")
plt.xlabel("frequency")
plt.ylabel("magnitude")

plt.subplot(3,1,3)
plt.plot(f,p)
plt.title("phase spectrum")
plt.xlabel("frequency")
plt.ylabel("phase")
plt.show()
