import numpy as np
from matplotlib import pyplot as plt

"""x1_n = np.array([1, 2, 3, 4, 5])
x2_n = np.array([1, 2, 3, 4, 5])

#Linearity
x3_n = np.array(x1_n + x2_n)
N1 = len(x1_n)
N2 = len(x2_n)
N3 = len(x3_n)

# Define the frequency range for the DTFT
omega = np.linspace(-np.pi, np.pi, 1000)

# Compute the DTFT
X1_omega = np.array([np.sum(x1_n * np.exp(-1j * k * np.arange(N1))) for k in omega])
X2_omega = np.array([np.sum(x2_n * np.exp(-1j * k * np.arange(N2))) for k in omega])
X3_omega = np.array([np.sum(x3_n * np.exp(-1j * k * np.arange(N3))) for k in omega])
x_omega = X2_omega + X1_omega
if (X3_omega.all() == x_omega.all()):
	print("Linearity test done")
else:
	print("Error")"""

"""#Time Shifting
x1_n = [1,2,3,0,0,0]
N1 = len(x1_n)
omega = np.linspace(-np.pi, np.pi, 1000)
x_w = np.array([np.sum(x1_n * np.exp(-1j * k * np.arange(N1))) for k in omega])
x_shift = np.roll(x1_n, 3)
X_theoretical = x_w * np.exp(-1j * omega * 3)
x_shift_w = np.array([np.sum(x_shift * np.exp(-1j * k * np.arange(N1))) for k in omega])
if (x_shift_w.all() == X_theoretical.all()):
	print("Time shifting test pass")
else:
	print("Error")"""

"""#Time reversal
x1_n = np.array([1, 2, 3, 4, 5])
omega = np.linspace(-np.pi, np.pi, 1000)
N1 = len(x1_n)
X1_omega = np.array([np.sum(x1_n * np.exp(-1j * k * np.arange(N1))) for k in omega])
x2_n = x1_n[::-1]
X2_omega = np.array([np.sum(x2_n * np.exp(-1j * k * np.arange(N1))) for k in omega])
x2_omega = X2_omega[::-1]
if (X1_omega.all() == x2_omega.all()):
	print("Time reversal test pass")
else:
	print("Error")"""

#Convolution
x1_n = np.array([1, 2, 3, 4, 5])
x2_n = np.array([1, 2, 3, 4, 5])
omega = np.linspace(-np.pi, np.pi, 1000)
X1_omega = np.array([np.sum(x1_n * np.exp(-1j * k * np.arange(5))) for k in omega])
X2_omega = np.array([np.sum(x2_n * np.exp(-1j * k * np.arange(5))) for k in omega])
x3_w = X1_omega.all() * X2_omega.all()
con = np.convolve(x1_n, x2_n, mode='full')
if(x3_w.all() == con.all()):
	print("convolution test done")
else:
	print("Error")





