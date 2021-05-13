# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
# importing Qiskit
from qiskit import *
from qiskit.visualization import plot_histogram

plt.style.use('seaborn-poster')
# get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Quantum FT
# 
# Next, we let's try to achieve the same result with the qft circuit

# %%
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
        circuit.barrier(qubit)
        circuit.measure_all()

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    measure(circuit,n)
    return circuit


# %%

def preprocessSignal(signal):
    signal = signal*SCALER
    signal = signal + abs(min(signal))

    return signal


# %%
#Parameters:
SCALER = 10

signal = preprocessSignal(x)
# x_processed = x_processed[2:4]
CIRCUIT_SIZE = int(max(signal)).bit_length() # this basically defines the "adc resolution"
print(f"Using Scaler {SCALER} resulted in Circuit Size {CIRCUIT_SIZE}")


# %%
# qc = QuantumCircuit(CIRCUIT_SIZE)
# qft(qc,CIRCUIT_SIZE)
# qc.draw()


# %%
def encodeInteger(circuit, integer):
    if integer.bit_length() > circuit.width():
        raise RuntimeError("Integer too large")

    binary = bin(integer)[2:]
    for i in range(0,min(len(binary),circuit.width()-1)):
        if binary[i] == '1':
            circuit.x(i)

    return circuit

def runCircuit(circuit):
    backend = Aer.get_backend("statevector_simulator")
    qobj = assemble(circuit)
    output = backend.run(qobj).result().get_statevector()
    
    return output
        
def decodeInteger(integer):
    return int(integer,2)

def accumulate(buffer, value):
    buffer[value] += 1

    return buffer

def processQFT(circuit, signal):
    signal_hat = np.zeros(signal.size)

    for i in range(0,signal.size):
        circuit = QuantumCircuit(CIRCUIT_SIZE, CIRCUIT_SIZE)
        circuit.reset(range(CIRCUIT_SIZE))

        circuit = encodeInteger(circuit, int(signal[i]))
        qft(circuit,CIRCUIT_SIZE)
        

        output = runCircuit(circuit)
        
        signal_hat = accumulate(signal_hat, output.argmax(axis=0))

        print(f"Processing index {i} with value {int(signal[i])} yielded {output.argmax(axis=0)}")

    return signal_hat


# %%
i=0
circuit = QuantumCircuit(CIRCUIT_SIZE, CIRCUIT_SIZE)
circuit.reset(range(CIRCUIT_SIZE))

circuit = encodeInteger(circuit, int(signal[i]))
qft(circuit,CIRCUIT_SIZE)

backend = Aer.get_backend("qasm_simulator")
job = execute(circuit, backend, shots=1, memory=True)
output = job.result().get_memory()[0]
# circuit.draw()


# %%
signal_hat = processQFT(circuit, signal)


# %%
n_oneside = N//2
# get the one side frequency
f_oneside = freq[:n_oneside]/(2*np.pi)

# normalize the amplitude
X_oneside =signal_hat[:n_oneside]/n_oneside

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(f_oneside, abs(X_oneside), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Normalized Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')

plt.subplot(122)
plt.stem(f_oneside, abs(X_oneside), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.xlim(0, 10)
plt.tight_layout()
plt.show()


# %%



