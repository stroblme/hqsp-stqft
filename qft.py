import time
from tkinter.constants import NO
from IPython import get_ipython

import datetime

import numpy as np
from numpy import array, pi

from math import log2

from qiskit import *
from qiskit.providers.aer.backends.aer_simulator import AerSimulator
from qiskit.circuit.library import QFT as qiskit_qft

import inspect
from qiskit.test import mock
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

from qiskit.circuit.library import QFT


from qiskit.tools.monitor import job_monitor

from utils import isPow2


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

def loadBackend(backendName, simulation=True):
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider("ibm-q")

    isMock = False
    try:
        backend = provider.get_backend(backendName)
    except QiskitBackendNotFoundError as e:
        print(f"Backend {backendName} not found in {provider}.\nTrying mock backend..")
        
        try:
            tempBackendModule = getattr(mock, backendName.replace("ibmq_", ''))
            backend = inspect.getmembers(tempBackendModule)[0][1]()
            isMock = True
        except QiskitBackendNotFoundError as e:
            print(f"Backend {backendName} also not found in mock devices. Check if the name is valid and has 'ibmq_' as prefix")
        except IndexError:
            print(f"Sorry, but mock backend didn't returned the expected structure. Check {tempBackendModule}")

    if not isMock:
        props = backend.properties(datetime=datetime.datetime.now())
    else:
        props = backend.properties()

    # backend = least_busy(  provider.backends(filters=lambda x: x.configuration().n_qubits >= 5
    #                             and not x.configuration().simulator
    #                             and x.status().operational==True))

    nQubitsAvailable = len(props.qubits)
    qubitReadoutErrors = [props.qubits[i][4].value for i in range(0, nQubitsAvailable)]
    qubitProbMeas0Prep1 = [props.qubits[i][5].value for i in range(0, nQubitsAvailable)]
    qubitProbMeas1Prep0 = [props.qubits[i][6].value for i in range(0, nQubitsAvailable)]

    print(f"Backend {backend} has {nQubitsAvailable} qubits available.")
    print(f"ReadoutErrors are {qubitReadoutErrors}")
    print(f"ProbMeas0Prep1 are {qubitProbMeas0Prep1}")
    print(f"ProbMeas1Prep0 are {qubitProbMeas1Prep0}")

    if simulation:
        backend = AerSimulator.from_backend(backend)

    return provider, backend

class qft_framework():
    # minRotation = 0.2 #in [0, pi/2)

    def __init__(self, numOfShots=2048, show=-1, minRotation=0, fixZeroSignal=False, suppressPrint=False, draw=False,
    simulation=True, backendName=None, reuseBackend=None):
        self.suppressPrint = suppressPrint
        self.show = show
        self.numOfShots = numOfShots
        self.minRotation = minRotation
        self.draw = draw

        self.simulation = simulation
        self.fixZeroSignal = fixZeroSignal        

        if reuseBackend != None:
            print(f"Reusing backend {reuseBackend}")
            self.backend = reuseBackend
        else:
            self.setBackend(backendName, simulation)

        self.mitigateResults = False
        self.measFitter = None

    def getBackend(self):
        return self.backend

    def setBackend(self, backendName=None, simulation=True):
        if backendName != None:
            self.provider, self.backend = loadBackend(backendName=backendName, simulation=simulation)
        else:
            if not self.simulation:
                print("Setting simulation to 'True', as no backend was specified")
                self.simulation = True
            self.backend = Aer.get_backend('qasm_simulator')

    def estimateSize(self, y_signal):
        assert isPow2(y_signal.nSamples)

        n_qubits = int((log2(y_signal.nSamples)/log2(2)))

        return 2**n_qubits

    def transform(self, y_signal, draw=False):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """
        self.samplingRate = y_signal.samplingRate
        y = y_signal.sample()

        y_hat = self.processQFT(y)

        return y_hat

    def transformInv(self, y_signal, draw=False):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """
        self.samplingRate = y_signal.samplingRate
        y_hat = y_signal.sample()

        y = self.processIQFT(y_hat)

        return y

    def postProcess(self, y_hat, f):
        y_hat, f = self.qubitNoiseFilter(y_hat=y_hat, f=f)


        return y_hat, f

    def loadMeasurementFitter(self, measFitter):
        self.measFitter = measFitter
        print(self.measFitter.cal_matrix)

        print(f"Enabling mitigating results from now on..")
        self.mitigateResults = True

    def setupMeasurementFitter(self, nQubits, nShots=1024):
        """In parts taken from https://quantumcomputing.stackexchange.com/questions/10152/mitigating-the-noise-in-a-quantum-circuit

        Args:
            nQubits ([type]): [description]
            nShots (int, optional): [description]. Defaults to 1024.
        """
        if self.backend is None:
            print("Need a backend first")
            self.measFitter = None
            return None

        measCalibrations, state_labels = complete_meas_cal(qr=QuantumRegister(nQubits), circlabel='mcal')

        print(f"Running measurement for filter on {nQubits} Qubits using {nShots} shots")
        job = execute(measCalibrations, backend=self.backend, shots=nShots)
        job_monitor(job, interval=5)
        cal_results = job.result()

        self.measFitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
        print(self.measFitter.cal_matrix)

        print(f"Enabling mitigating results from now on..")
        self.mitigateResults = True

        return self.measFitter

    def qubitNoiseFilter(self, jobResult):
        """In parts taken from https://quantumcomputing.stackexchange.com/questions/10152/mitigating-the-noise-in-a-quantum-circuit

        Args:
            jobResult ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.measFitter == None:
            print("Need to initialize measurement fitter first")
            return jobResult

        # Get the filter object
        measFilter = self.measFitter.filter

        # Results with mitigation
        mitigatedResult = measFilter.apply(jobResult)
        # mitigatedCounts = mitigatedResult.get_counts(0)
        print(f"Filtering achieved at '0000': {jobResult.get_counts()['0000']} vs before: {mitigatedResult.get_counts()['0000']}")
        return mitigatedResult

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
        circuit.h(n) # apply hadamard
        
        rotGateSaveCounter = 0

        for qubit in range(n):
            rot = pi/2**(n-qubit)
            if rot < self.minRotation:
                rotGateSaveCounter += 1
                if not self.suppressPrint:
                    print(f"Rotations lower than {self.minRotation}: is {rot}")
            else:
                circuit.cp(rot, qubit, n)

        # At the end of our function, we call the same function again on
        # the next qubits (we reduced n by one earlier in the function)
        if n != 0 and rotGateSaveCounter != 0 and not self.suppressPrint:
            print(f"Saved {rotGateSaveCounter} rotation gates which is {int(100*rotGateSaveCounter/n)}% of {n} qubits")
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


    def processQFT(self, y):

        """
        Args:
        amplitudes: List - A list of amplitudes with length equal to power of 2
        normalize: Bool - Optional flag to control normalization of samples, True by default
        Returns:
        circuit: QuantumCircuit - a quantum circuit initialized to the state given by amplitudes
        """
        n_samples = y.size
        assert isPow2(n_samples)

        n_qubits = int((log2(n_samples)/log2(2)))
        if not self.suppressPrint:
            print(f"Using {n_qubits} Qubits to encode {n_samples} Samples")     

        if y.max() == 0.0:
            if self.fixZeroSignal:
                print(f"Warning: Signal's max value is zero and therefore amplitude initialization will fail. Setting signal to constant-one to continue")
                y = np.ones(n_samples)
            else:
                y_hat = np.zeros(2**n_qubits)
                return y_hat

        q = QuantumRegister(n_qubits,'q')
        qc = QuantumCircuit(q)

        # Normalize ampl, which is required for squared sum of amps=1
        ampls = y / np.linalg.norm(y)

        # for 2^n amplitudes, we have n qubits for initialization
        # this means that the binary representation happens exactly here

        qc.initialize(ampls, [q[i] for i in range(n_qubits)])
        # qc += QFT(num_qubits=n_qubits, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name='qft')

        qc = self.qft(qc, n_qubits)
        qc.measure_all()
        
        if self.draw:
            self.draw=False
            name = str(time.mktime(datetime.datetime.now().timetuple()))[:-2]
            qc.draw(output='mpl', filename=f'./export/{name}.png')

        

        if not self.suppressPrint:
            print(f"Transpiling for {self.backend}")
    
        qc = transpile(qc, self.backend, optimization_level=1) # opt level 0,1..3. 3: heaviest opt

        if not self.suppressPrint:
            print("Executing job...")
    
        #substitute with the desired backend
        job = execute(qc, self.backend, shots=self.numOfShots)
        # if job.status != "COMPLETED":
        job_monitor(job, interval=5) #run a blocking monitor thread

        if self.mitigateResults:
            jobResult = self.qubitNoiseFilter(job.result())
        else:
            print("Warning: Mitigatin results is implicitly disabled. Consider enabling it by running 'setupMeasurementFitter'")
            jobResult = job.result()

        counts = jobResult.get_counts()
        y_hat = np.array(get_fft_from_counts(counts, n_qubits))
        # [:n_samples//2]
        y_hat = self.dense(y_hat, D=max(n_qubits/(self.samplingRate/n_samples),1))
        # top_indices = np.argsort(-np.array(fft))
        # freqs = top_indices*self.samplingRate/n_samples
        # get top 5 detected frequencies

        
        return y_hat

    def processIQFT(self, y):
        n_samples = y.size
        assert isPow2(n_samples)

        n_qubits = int((log2(n_samples)/log2(2)))
        if not self.suppressPrint:
            print(f"Using {n_qubits} Qubits to encode {n_samples} Samples")     

        if y.max() == 0.0:
            y_hat = np.zeros(2**n_qubits)
            return y_hat

        q = QuantumRegister(n_qubits)
        qc = QuantumCircuit(q)

        # Normalize ampl, which is required for squared sum of amps=1
        ampls = y / np.linalg.norm(y)

        # for 2^n amplitudes, we have n qubits for initialization
        # this means that the binary representation happens exactly here
        qc.initialize(ampls, [q[i] for i in range(n_qubits)])

        qc = self.qft(qc, n_qubits)
        qc.inverse()
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

        if self.draw:
            self.draw=False
            name = str(time.mktime(datetime.datetime.now().timetuple()))[:-2]
            qc.draw(output='mpl', filename=f'./export/{name}.png')
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
