import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
# importing Qiskit
from qiskit import *
from qiskit.visualization import plot_histogram

plt.style.use('seaborn-poster')


def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    
    return X


# sampling rate
sr = 100
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)

plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')
plt.xlabel('Normalized Time')

plt.show()

X = DFT(x)

# calculate the frequency
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

n_oneside = N//2
# get the one side frequency
f_oneside = freq[:n_oneside]

# normalize the amplitude
X_oneside =X[:n_oneside]/n_oneside

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(f_oneside, abs(X_oneside), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')

plt.subplot(122)
plt.stem(f_oneside, abs(X_oneside), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.xlim(0, 10)
plt.tight_layout()
plt.show()



def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    
    
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def measure(circuit, n):
    for qubit in range(n):
        circuit.measure(qubit, 0)

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    measure(circuit,n)
    return circuit

SCALER = 1
def preprocessSignal(signal):
    signal = signal*SCALER
    signal = signal + abs(min(signal))

    return signal

#Parameters:

signal = preprocessSignal(x)
# x_processed = x_processed[2:4]
CIRCUIT_SIZE = int(max(signal)).bit_length() # this basically defines the "adc resolution"
print(CIRCUIT_SIZE)

def encodeInteger(encIntegerCircuit, integer):
    if integer.bit_length() > encIntegerCircuit.width():
        raise RuntimeError("Integer too large")

    binary = bin(integer)[2:]
    for i in range(0,min(len(binary),encIntegerCircuit.width()-1)):
        if binary[i] == '1':
            encIntegerCircuit.x(i)

    return encIntegerCircuit

def processQFT(circuit, signal):
    result = list()

    for i in range(0,signal.size):
        circuit = QuantumCircuit(CIRCUIT_SIZE, 1)
        circuit.reset(range(CIRCUIT_SIZE))

        qft(circuit,CIRCUIT_SIZE)
        circuit = encodeInteger(circuit, int(signal[i]))

        backend = Aer.get_backend("qasm_simulator")
        job = execute(circuit, backend, shots=1, memory=True)
        output = job.result().get_memory()[0]

        print(f"Processing index {i} with value {int(signal[i])} yielded {output}")

circuit = processQFT(circuit, signal)
