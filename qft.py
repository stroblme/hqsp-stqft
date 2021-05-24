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

    def transform(self, y_signal, show=-1):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """        
        self.samplingRate = y_signal.samplingRate
        y = y_signal.sample()

        print(f"Stretching signal with scalar {self.scaler}")
        y_preprocessed = self.preprocessSignal(y, self.scaler)

        # x_processed = x_processed[2:4]
        print(f"Calculating required qubits for encoding a max value of {int(max(y_preprocessed))}")
        circuit_size = int(max(y_preprocessed)).bit_length() # this basically defines the "adc resolution"

        # y_hat = self.processQFT_dumb(y_preprocessed, circuit_size, show)
        y_hat = self.processQFT_layerwise(y_preprocessed, circuit_size, show)
        # y_hat = self.processQFT_geometric(y_preprocessed, circuit_size, show)
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
        results = backend.run(qobj).result()
        output = results.get_statevector()
        
        return output
            
    def decode(self, buffer, value):
        # buffer = value.argmax(axis=0)
        buffer = self.accumulate(buffer, value.argmax(axis=0))
        # buffer = self.accumulate(buffer, int(bin(value.argmax(axis=0)),2))

        return buffer

    def accumulate(self, buffer, value, intense=1):
        buffer[value] += intense

        return buffer


    def dense(self, y_hat, D=3):
        
        y_hat_densed = np.zeros(int(y_hat.size/D))

        for i in range(y_hat.size-1):
            # if i%D != 0:
            y_hat_densed[int(i/D)] += abs(y_hat[i])

        return y_hat_densed

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

    def processQFT_geometric(self, y, circuit_size, show=-1):
        y_hat = np.zeros(y.size)

        print(f"Generating circuit consisting of {circuit_size} qubits")

        qreg_q = QuantumRegister(circuit_size, 'q')
        creg_c = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.reset(range(circuit_size))
        print(f"Encoding {y.size} input values")
        # circuit = self.qft(circuit,circuit_size)
        
        for i in range(0,y.size):
            circuit = self.encode(circuit, int(y[i]))


            # self.iqft(circuit,circuit_size)

            circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=True, name='qft')
            circuit.measure(qreg_q[0], creg_c) #measure first or last one?
            # circuit.measure(qreg_q[circuit_size-1], creg_c) #measure first or last one?

            output = self.runCircuit(circuit)
            
            y_hat 

            # print(f"Processing index {i} with value {int(y[i])} yielded {output.argmax(axis=0)}")
            if show!=-1 and i==show:
                circuit.draw('mpl', style='iqx')
                return None

        return y_hat

    def processQFT_layerwise(self, y, circuit_size, show=-1):
        #https://algassert.com/quirk#circuit={%22cols%22:[[{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%20t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%20t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%20t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%20t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%20t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%20t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%20t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%20t)%20+%20pi/4%22}],[],[],[],[],[{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%202t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%202t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%202t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%202t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%202t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%202t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%202t)%20+%20pi/4%22},{%22id%22:%22Rxft%22,%22arg%22:%221/2%20sin(pi%202t)%20+%20pi/4%22}],[],[],[],[],[%22Bloch%22,%22Bloch%22,%22Bloch%22,%22Bloch%22,%22Bloch%22,%22Bloch%22,%22Bloch%22,%22Bloch%22],[%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22],[%22Swap%22,1,1,1,1,1,1,%22Swap%22],[1,%22Swap%22,1,1,1,1,%22Swap%22],[1,1,%22Swap%22,1,1,%22Swap%22],[1,1,1,%22Swap%22,%22Swap%22],[%22H%22],[%22Z^%C2%BD%22,%22%E2%80%A2%22],[1,%22H%22],[%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],[1,1,%22H%22],[%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],[1,1,1,%22H%22],[%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],[1,1,1,1,%22H%22],[%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],[1,1,1,1,1,%22H%22],[%22Z^%E2%85%9F%E2%82%86%E2%82%84%22,%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],[1,1,1,1,1,1,%22H%22],[%22Z^%E2%85%9F%E2%82%81%E2%82%82%E2%82%88%22,%22Z^%E2%85%9F%E2%82%86%E2%82%84%22,%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],[1,1,1,1,1,1,1,%22H%22],[%22Chance8%22]]}
        # circuit = self.qft(circuit,circuit_size)
        # circuit = self.encode(circuit, int(y[0]))
        batchSize = 20  # This inherently defines the number of qubits
        if self.samplingRate%batchSize != 0:
            raise ValueError(f"Sampling Rate {self.samplingRate} must be a multiple of the batch size {batchSize}")

        absMaxY = max(y.max(), abs(y.min()))
        sizeY = y.size

        output_vector = None
        batches = int(self.samplingRate/batchSize)
        circuit_size = int(sizeY/batches)

        print(f"Using {batches} batches with a size of {batchSize}")

        THETA_RANGE = np.pi/2 * 0.2

        for b in range(1, sizeY, circuit_size): #from 1 to sizeY in steps of circuit_size
             
            qreg_q = QuantumRegister(circuit_size, 'q')
            creg_c = ClassicalRegister(1, 'c')
            circuit = QuantumCircuit(qreg_q, creg_c)
            circuit.reset(range(circuit_size))

            for i in range(0, circuit_size, 1): #explicit step size to prevent confusion
                idx = b+i-1             #normal b+i-1 is [0,..,ciruit_size*sizeY]
                # idx = sizeY-1-(b+i-1) #inverse

                theta = THETA_RANGE*y[idx]/absMaxY+np.pi/2 #is [0,..,PI/2) for y<0 and [PI/2,..,2PI) for y<0
                # print(b+i-1)
                # circuit.x(qreg_q[i])
                print(theta)
                circuit.rx(theta,qreg_q[i])
                circuit.barrier(qreg_q[i])
                # circuit.cx()

            # circuit = self.qft(circuit,circuit_size)
            circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft')
            # circuit.measure(qreg_q[circuit_size-1], creg_c) #measure first or last one?
            circuit.measure(qreg_q[0], creg_c) #measure first or last one?
            # circuit.measure_all()

            result = self.runCircuit(circuit)
            if b==1:
                output_vector = np.array(result)

            else:
                output_vector += np.array(result)


            # self.iqft(circuit,circuit_size)

            # circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft')
            # circuit += qiskit_qft(num_qubits=circuit_size, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=True, name='qft')


        # circuit.draw('mpl', style='iqx')

        return self.dense(output_vector)


        # 1. iterative encode but use results from prev. qft
