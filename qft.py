from IPython import get_ipython

import numpy as np
from numpy import array, pi

from math import log2

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT as qiskit_qft

def isPow2(x):
    return (x!=0) and (x & (x-1)) == 0


def get_bit_string(n, n_qubits):
    """
    Returns the binary string of an integer with n_qubits characters
    """

    assert n < 2**n_qubits, 'n too big to binarise, increase n_qubits or decrease n'

    bs = "{0:b}".format(n)
    bs = "0"*(n_qubits - len(bs)) + bs

    return bs

def get_fft_from_counts(counts, n_qubits):

    out = []
    keys = counts.keys()
    for i in range(2**n_qubits):
        id = get_bit_string(i, n_qubits)
        if(id in keys):
            out.append(counts[id])
        else:
            out.append(0)

    return out

class qft_framework():
    def __init__(self, numOfShots=1024):
        self.numOfShots = numOfShots
        pass

    def transform(self, y_signal, show=-1):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """
        self.samplingRate = y_signal.samplingRate
        y = y_signal.sample()

        # print(f"Stretching signal with scalar {self.scaler}")
        # y_preprocessed = self.preprocessSignal(y, self.scaler)

        # x_processed = x_processed[2:4]
        # print(f"Calculating required qubits for encoding a max value of {int(max(y_preprocessed))}")
        # circuit_size = int(max(y_preprocessed)).bit_length() # this basically defines the "adc resolution"

        # y_hat = self.processQFT_dumb(y_preprocessed, circuit_size, show)
        # y_hat = self.processQFT_layerwise(y_preprocessed, circuit_size, show)
        y_hat = self.processQFT_schmidt(y)
        # y_hat = self.processQFT_geometric(y_preprocessed, circuit_size, show)
        return y_hat

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
        # self.measure(circuit,n)
        return circuit

    def inverseQft(self, circuit, n):
        """Inverse QFT on the first n qubits in the circuit"""
        q_circuit = self.qft(QuantumCircuit(n), n)
        inv_q_ciruit = q_circuit.inverse()
        circuit.append(inv_q_ciruit, circuit.qubits[:n])

        return circuit.decompose()


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
        if D==0:
            return y_hat
        D = int(D)
        y_hat_densed = np.zeros(int(y_hat.size/D))

        for i in range(y_hat_densed.size*D-1):
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


    def processQFT_schmidt(self, samples):

        """
        Args:
        amplitudes: List - A list of amplitudes with length equal to power of 2
        normalize: Bool - Optional flag to control normalization of samples, True by default
        Returns:
        circuit: QuantumCircuit - a quantum circuit initialized to the state given by amplitudes
        """
        n_samples = len(samples)
        assert isPow2(n_samples)

        n_qubits = int((log2(n_samples)/log2(2)))
        print(f"Using {n_qubits} Qubits to encode {n_samples} Samples")        
        q = QuantumRegister(n_qubits)
        qc = QuantumCircuit(q)

        # Normalize ampl, which is required for squared sum of amps=1
        ampls = samples / np.linalg.norm(samples)

        # for 2^n amplitudes, we have n qubits for initialization
        # this means that the binary representation happens exactly here
        qc.initialize(ampls, [q[i] for i in range(n_qubits)])

        qc = self.qft(qc, n_qubits)
        qc.measure_all()

        qasm_backend = Aer.get_backend('qasm_simulator')
        # real_backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 5
        #                                        and not x.configuration().simulator
        #                                        and x.status().operational==True))


        #substitute with the desired backend
        out = execute(qc, qasm_backend, shots=self.numOfShots).result()
        counts = out.get_counts()
        y_hat = np.array(get_fft_from_counts(counts, n_qubits))
        # [:n_samples//2]
        y_hat = self.dense(y_hat, D=max(n_qubits/(self.samplingRate/n_samples),1))
        # top_indices = np.argsort(-np.array(fft))
        # freqs = top_indices*self.samplingRate/n_samples
        # get top 5 detected frequencies


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
