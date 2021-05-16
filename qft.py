from IPython import get_ipython

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt

from qiskit import *
from qiskit.visualization import plot_histogram

plt.style.use('seaborn-poster')




class qft_framework():
    def __init__(self) -> None:
        self.setScaler()

    def transform(self, y):
        print(f"Stretching signal with scalar {self.scaler}")
        y_preprocessed = self.preprocessSignal(y, self.scaler)

        # x_processed = x_processed[2:4]
        print(f"Calculating required qubits for encoding a max value of {int(max(y_preprocessed))}")
        circuit_size = int(max(y_preprocessed)).bit_length() # this basically defines the "adc resolution"

        y_hat = self.processQFT(y_preprocessed, circuit_size)
        return y_hat

    def setScaler(self, scaler=10):
        self.scaler = scaler

    def showCircuit(self, y):
        print(f"Stretching signal with scalar {self.scaler}")
        y_preprocessed = self.preprocessSignal(y, self.scaler)

        # x_processed = x_processed[2:4]
        print(f"Calculating required qubits for encoding a max value of {int(max(y_preprocessed))}")
        circuit_size = int(max(y_preprocessed)).bit_length() # this basically defines the "adc resolution"


        print(f"Generating circuit consisting of {circuit_size} qubits")
        circuit = QuantumCircuit(circuit_size, circuit_size)
        self.qft(circuit,circuit_size)
        circuit.reset(range(circuit_size))

        circuit = self.encodeInteger(circuit, int(y))

        circuit.draw()

    def qft_rotations(self, circuit, n):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(pi/2**(n-qubit), qubit, n)
        
        
        # At the end of our function, we call the same function again on
        # the next qubits (we reduced n by one earlier in the function)
        self.qft_rotations(circuit, n)

    def swap_registers(self, circuit, n):
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        return circuit

    def measure(self, circuit, n):
        for qubit in range(n):
            circuit.barrier(qubit)
        circuit.measure_all()

    def qft(self, circuit, n):
        """QFT on the first n qubits in circuit"""
        self.qft_rotations(circuit, n)
        self.swap_registers(circuit, n)
        self.measure(circuit,n)
        return circuit

    def preprocessSignal(self, y, scaler):
        '''
        Preprocessing signal using a provided scaler
        '''

        y = y*scaler
        y = y + abs(min(y))

        return y

    def encodeInteger(self, circuit, integer):
        if integer.bit_length() > circuit.width():
            raise RuntimeError("Integer too large")

        binary = bin(integer)[2:]
        for i in range(0,min(len(binary),circuit.width()-1)):
            if binary[i] == '1':
                circuit.x(i)

        return circuit

    def runCircuit(self, circuit):
        backend = Aer.get_backend("statevector_simulator")
        qobj = assemble(circuit)
        output = backend.run(qobj).result().get_statevector()
        
        return output
            
    # def decodeInteger(buffer, integer):
    #     return int(integer,2)

    def accumulate(self, buffer, value):
        buffer[value] += 1

        return buffer

    def processQFT(self, y, circuit_size):
        y_hat = np.zeros(y.size)

        print(f"Generating circuit consisting of {circuit_size} qubits")
        circuit = QuantumCircuit(circuit_size, circuit_size)
        self.qft(circuit,circuit_size)
        circuit.reset(range(circuit_size))

        print(f"Encoding {y.size} input values")
        for i in range(0,y.size):

            circuit = self.encodeInteger(circuit, int(y[i]))
            

            output = self.runCircuit(circuit)
            
            # y_hat = accumulate(y_hat, output.argmax(axis=0))
            y_hat = self.accumulate(y_hat, int(bin(output.argmax(axis=0)),2))

            # print(f"Processing index {i} with value {int(y[i])} yielded {output.argmax(axis=0)}")

        return y_hat