from IPython import get_ipython

import numpy as np
from numpy import array, pi

import matplotlib.pyplot as plt

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT as qiskit_qft

plt.style.use('seaborn-poster')




class qft_framework():
    def __init__(self) -> None:
        self.setScaler()

    def transform(self, y, show=-1):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """        
        print(f"Stretching signal with scalar {self.scaler}")
        y_preprocessed = self.preprocessSignal(y, self.scaler)

        # x_processed = x_processed[2:4]
        print(f"Calculating required qubits for encoding a max value of {int(max(y_preprocessed))}")
        circuit_size = int(max(y_preprocessed)).bit_length() # this basically defines the "adc resolution"

        # y_hat = self.processQFT_dumb(y_preprocessed, circuit_size, show)
        y_hat = self.processQFT_layerwise(y_preprocessed, circuit_size, show)
        return y_hat

    def setScaler(self, scaler=1):
        self.scaler = scaler

    def showCircuit(self, y):
        """Display the circuit for a signal y

        Args:
            y (signal): signal instance used for circuit configuration
        """        
        self.transform(y,int(y.size / 3))


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

    def preprocessSignal(self, y, scaler, shift=False):
        '''
        Preprocessing signal using a provided scaler
        '''

        y = y*scaler
        if shift:
            y = y + abs(min(y))

        return y

    def encode(self, circuit, integer):
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
            
    def decode(self, buffer, value):
        # buffer = value.argmax(axis=0)
        buffer = self.accumulate(buffer, value.argmax(axis=0))
        # buffer = self.accumulate(buffer, int(bin(value.argmax(axis=0)),2))

        return buffer

    def accumulate(self, buffer, value):
        buffer[value] += 1

        return buffer

    def processQFT_dumb(self, y, circuit_size, show=-1):
        y_hat = np.zeros(y.size)

        print(f"Generating circuit consisting of {circuit_size} qubits")

        circuit = QuantumCircuit(circuit_size, circuit_size)
        circuit.reset(range(circuit_size))
        print(f"Encoding {y.size} input values")
        circuit = self.qft(circuit,circuit_size)
        
        for i in range(0,y.size):
            circuit = self.encode(circuit, int(y[i]))


            # self.iqft(circuit,circuit_size)

            # circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft')
            # circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=True, name='qft')

            output = self.runCircuit(circuit)
            
            y_hat = self.decode(y_hat, output)

            # print(f"Processing index {i} with value {int(y[i])} yielded {output.argmax(axis=0)}")
            if show!=-1 and i==show:
                circuit.draw('mpl', style='iqx')
                return None

        return y_hat

    def dense(self, y_hat, D=3):
        
        y_hat_densed = np.zeros(int(y_hat.size/D))

        for i in range(y_hat.size-1):
            if i%D != 0:
                y_hat_densed[int(i/D)] += y_hat[i]

        return y_hat_densed


    def processQFT_layerwise(self, y, circuit_size, show=-1):
        y_hat = np.zeros(y.size)
        maxY = y.max()

        print(f"Generating circuit consisting of {circuit_size} qubits")

        
        # circuit = self.qft(circuit,circuit_size)
        # circuit = self.encode(circuit, int(y[0]))
        output_vector = None
        batches = 4    # the higher the less qubits
        circuit_size = int(y.size/batches)

        for b in range(1, y.size, circuit_size):
            qreg_q = QuantumRegister(circuit_size, 'q')
            creg_c = ClassicalRegister(1, 'c')
            circuit = QuantumCircuit(qreg_q, creg_c)
            circuit.reset(range(circuit_size))

            for i in range(0,circuit_size):
                theta = 2*np.pi*y[b+i-1]/maxY # (i+1)*b-1 is [0,..,]
                print(b+i-1)
                circuit.rx(2*np.pi*y[b+i-1]/maxY,qreg_q[i])
            # circuit = self.qft(circuit,circuit_size)
            circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft')
            # circuit.measure(qreg_q[circuit_size-1], creg_c) #measure first or last one?
            circuit.measure_all()

            if b==1:
                output_vector = np.array(self.runCircuit(circuit))

            else:
                output_vector += np.array(self.runCircuit(circuit))


            # self.iqft(circuit,circuit_size)

            # circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft')
            # circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=True, name='qft')


        circuit.draw('mpl', style='iqx')

        return self.dense(output_vector)


        # 1. iterative encode but use results from prev. qft
