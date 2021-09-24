import time

import datetime

import numpy as np
from numpy import pi

import copy

from math import log2

from qiskit import *
from qiskit.providers.aer.backends.aer_simulator import AerSimulator
from qiskit.providers.aer import noise
# from qiskit.circuit.library import QFT as qiskit_qft

import inspect
# from qiskit.providers.aer.noise import noise_model
from qiskit.test import mock
from qiskit.providers.exceptions import QiskitBackendNotFoundError
# from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

# from qiskit.circuit.library import QFT

from qiskit.tools.monitor import job_monitor

from frontend import signal
# import mitiq

from utils import filterByThreshold, isPow2
import ibmAccounts #NOT redundant! needed to get account information! Can be commented out if loading e.g. noise data is not needed

def get_bit_string(n, n_qubits):
    """Returns the binary string of an integer with n_qubits characters

    Args:
        n (int): integer to be converted
        n_qubits (int): number of qubits

    Returns:
        string: binary string
    """

    assert n < 2**n_qubits, 'n too big to binarise, increase n_qubits or decrease n'

    bs = "{0:b}".format(n)
    bs = "0"*(n_qubits - len(bs)) + bs

    return bs

def hexKeyToBin(counts, n_qubits):
    """Generates binary representation of a hex based counts

    Args:
        counts (dict): dictionary with hex keys
        n_qubits (int): number of qubits

    Returns:
        dict: dictionary with bin keys
        n_qubits (int): number of qubits
    """
    out = dict()
    for key, value in counts.items():
        out[format(int(key,16), f'0{int(n_qubits)}b')] = value
    return out, n_qubits

def get_fft_from_counts(counts, n_qubits):
    """Calculates the fft based on the counts of an experiment

    Args:
        counts (int): dictionary with binary keys
        n_qubits (int): number of qubits

    Returns:
        dict: fft counts
    """
    out = []
    keys = counts.keys()
    for i in range(2**n_qubits):
        id = get_bit_string(i, n_qubits)
        if(id in keys):
            out.append(counts[id])
        else:
            out.append(0)

    return out

def loadBackend(backendName:str, simulation:bool=True, suppressPrint:bool=True):
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

    try:
        qubitReadoutErrors = [props.qubits[i][4].value for i in range(0, nQubitsAvailable)]
        qubitProbMeas0Prep1 = [props.qubits[i][5].value for i in range(0, nQubitsAvailable)]
        qubitProbMeas1Prep0 = [props.qubits[i][6].value for i in range(0, nQubitsAvailable)]

        if not suppressPrint:
            print(f"Backend {backend} has {nQubitsAvailable} qubits available.")
            print(f"ReadoutErrors are {qubitReadoutErrors}")
            print(f"ProbMeas0Prep1 are {qubitProbMeas0Prep1}")
            print(f"ProbMeas1Prep0 are {qubitProbMeas1Prep0}")
    except IndexError:
        print(f"Failed to get some properties. This can mean that they are simply not stored together with the mock backend")

    if simulation:
        backend = AerSimulator.from_backend(backend)

    return provider, backend

class qft_framework():
    # minRotation = 0.2 #in [0, pi/2)

    def __init__(self,  numOfShots:int=2048,
                        minRotation:int=0, signalThreshold:int=0, fixZeroSignal:bool=False, 
                        suppressPrint:bool=False, draw:bool=False,
                        simulation:bool=True,
                        noiseMitigationOpt:int=0, useNoiseModel:bool=False, backend=None, 
                        transpileOnce:bool=False, transpOptLvl:int=1):
                        
        self.suppressPrint = suppressPrint
        self.numOfShots = numOfShots
        self.minRotation = minRotation
        self.draw = draw

        # check if provided parameters are usefull
        if fixZeroSignal and signalThreshold > 0:
            print("Signal Filter AND zero fixer are enabled. This might result in a wasteful transform. Consider disabling Zero Fixer if not needed.")
        # transfer parameter
        self.fixZeroSignal = fixZeroSignal  
        self.signalThreshold = signalThreshold

        # transfer parameter
        self.transpOptLvl = transpOptLvl      

        # The following code will set:
        # backend
        # provider
        # simulation
        # noiseModelBackend
        # -------------------------------------------------

        # no backend -> simulation only
        if backend == None:
            # no simulation without a valid backend doesn't make sense
            if not simulation:
                print("Simulation was disabled but no backend provided. Will enable simulation")
            self.simulation = True
            if useNoiseModel:
                print("Noise model can be used without a corresponding backend")
            self.noiseModel = None

            self.backend = None
            self.provider = None
        # user provided backend only as a name, not instance
        elif type(backend) == str:
            # check if noise model should be used
            if useNoiseModel:
                # check if simulation was disabled
                if not simulation:
                    print("Simulation was disabled but backend provided and noise model enabled. Will enable simulation")
                self.simulation = True

                # set the noise model but do only load the simulator backend. Careful! IBMQ has a request limit ;)
                self.provider, tempBackend = loadBackend(backendName=backend, simulation=True)
                # generate noise model from backend properties
                self.noiseModel = noise.NoiseModel.from_backend(tempBackend)
                self.backend = self.getSimulatorBackend()

            else:
                # check if simulation was enabled
                if simulation:
                    print("Simulation was enabled but backend provided and noise model disabled. Will disable simulation")
                self.simulation = False

                # Null the noise model and load a backend for simulation or real device
                self.noiseModel = None
                self.provider, self.backend = loadBackend(backendName=backend, simulation=self.simulation)

        # user provided full backend instance
        else:
            # check if user provided a noise model
            if useNoiseModel:
                if not simulation:
                    print("Simulation was disabled but backend provided and noise model enabled. Will enable simulation")
                self.simulation = True

                # generate noise model from backend properties
                self.noiseModel = noise.NoiseModel.from_backend(backend)
                # and set the backend as simulator
                self.backend = self.getSimulatorBackend()
                self.provider = None
            else:
                if simulation:
                    print("Simulation was enabled but backend provided and noise model disabled. Will disable simulation")
                self.simulation = False

                self.noiseModel = None
                self.provider = None #TODO: check if this will cause problems
                self.backend = backend

        # -------------------------------------------------
        
        # transfer parameter
        self.noiseMitigationOpt = noiseMitigationOpt
        self.filterBackend = self.backend
        # # separate backend for noise filter provided?
        # if filterBackend == None:
        #     if self.suppressNoise and self.simulation and useNoiseModel:
        #         print("Warning this might will lead an key error later in transform, as simulation has no noise but noise model was enabled and no filter backend provided")
        #     self.filterBackend = self.backend
        # else:
        #     # check if noise model should be used
        #     if not useNoiseModel:
        #         self.provider, tempBackend = loadBackend(filterBackend, True)
        #         self.filterBackend = tempBackend
        #     else:
        #         self.provider, tempBackend = loadBackend(backend, True)
        #         self.noiseModelBackend = noise.NoiseModel.from_backend(tempBackend)
        #         self.filterBackend = self.getSimulatorBackend()


        # noise mitigation
        self.measFitter = None
        self.filterResultCounts = None
        self.customFilter = True    #TODO: rework such that we can choose a mitigation approach

        # transpilation reuse
        self.transpileOnce=transpileOnce
        self.transpiled = False
        

    def getBackend(self):
        """returns the current backend

        Returns:
            qiskit backend: backend used in qft
        """
        return self.backend

    def getSimulatorBackend(self):
        return Aer.get_backend('qasm_simulator')

    # def setBackend(self, backendName=None, simulation=True):
    #     if backendName != None:
    #         self.provider, self.backend = loadBackend(backendName=backendName, simulation=simulation)
    #     else:
    #         if not self.simulation:
    #             print("Setting simulation to 'True', as no backend was specified")
    #             self.simulation = True
    #         self.backend = self.getSimulatorBackend()

    def estimateSize(self, y_signal:signal):
        assert isPow2(y_signal.nSamples)

        n_qubits = int((log2(y_signal.nSamples)/log2(2)))

        return 2**n_qubits

    def transform(self, y_signal:signal):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """
        self.samplingRate = y_signal.samplingRate
        y = y_signal.sample()

        if self.signalThreshold > 0:
            # rm when eval done
            y = filterByThreshold(y, self.signalThreshold)

        y_hat = self.processQFT(y)

        return y_hat

    def transformInv(self, y_signal:signal):
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

    def loadMeasurementFitter(self, measFitter):
        self.measFitter = measFitter
        print(self.measFitter.cal_matrix)

        if self.noiseMitigationOpt != 1:
            print(f"Enabling noise mitigating option 1 from now on..")
            self.noiseMitigationOpt = 1

    def setupMeasurementFitter(self, nQubits:int, nShots:int, nRuns:int=10):
        """In parts taken from https://quantumcomputing.stackexchange.com/questions/10152/mitigating-the-noise-in-a-quantum-circuit

        Args:
            nQubits ([type]): [description]
            nShots (int, optional): [description]. Defaults to 1024.
        """
        if self.backend is None:
            print("Need a backend first")
            self.measFitter = None
            return None

        if self.customFilter:
            y = np.ones(2**nQubits)
            ampls = y / np.linalg.norm(y)

            q = QuantumRegister(nQubits,'q')
            qc = QuantumCircuit(q,name="noise mitigation circuit")

            qc.initialize(ampls, [q[i] for i in range(nQubits)])
            qc = self.qft(qc, nQubits)
            qc.measure_all()
            qc = transpile(qc, self.filterBackend, optimization_level=self.transpOptLvl) # opt level 0,1..3. 3: heaviest opt

            print(f"Running noise measurement {nRuns} times on {nQubits} Qubits with {nShots} shots.. This might take a while")

            jobResults = list()
            for n in range(nRuns):
                job = execute(qc, self.filterBackend, noise_model=self.noiseModel, shots=self.numOfShots)
                if not self.suppressPrint:
                    job_monitor(job, interval=5) #run a blocking monitor thread
                jobResult = job.result()
                jobResults.append(jobResult.results[0].data.counts)

            self.filterResultCounts = dict()
            for result in jobResults:
                self.filterResultCounts = {k: self.filterResultCounts.get(k, 0) + 1/nRuns*result.get(k, 0) for k in set(self.filterResultCounts) | set(result)}
            print(f"Filter Results: {self.filterResultCounts}")

        # else:
        #     measCalibrations, state_labels = complete_meas_cal(qr=QuantumRegister(nQubits), circlabel='mcal')

        #     print(f"Running measurement for filter on {nQubits} Qubits using {nShots} shots")
        #     job = execute(measCalibrations, backend=self.backend, shots=nShots)
        #     job_monitor(job, interval=5)
        #     cal_results = job.result()

        #     self.measFitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
        #     print(self.measFitter.cal_matrix)

        if self.noiseMitigationOpt != 1:
            print(f"Enabling noise mitigating option 1 from now on..")
            self.noiseMitigationOpt = 1

        return self.measFitter

    def qubitNoiseFilter(self, jobResult, nQubits:int):
        """In parts taken from https://quantumcomputing.stackexchange.com/questions/10152/mitigating-the-noise-in-a-quantum-circuit

        Args:
            jobResult ([type]): [description]

        Returns:
            [type]: [description]
        """
        # TODO: implement filter selection here
        if self.customFilter:
            if self.filterResultCounts == None:
                print("Need to initialize measurement fitter first")
                if nQubits == None:
                    print(f"For auto-initialization, you must provide the number of qubits")
                    return jobResult
                self.setupMeasurementFitter(nQubits=nQubits, nShots=jobResult.results[0].shots)
            elif len(self.filterResultCounts) == 1:
                print("Seems like you try to mitigate noise of a simulation without any noise. You can either disable noise suppression or consider running with noise.")
                return jobResult
            
            
            mitigatedResult = copy.deepcopy(jobResult)

            jobResultCounts = jobResult.results[0].data.counts

            maxCount = max(jobResultCounts.values()) #get max. number of counts in the plot

            nMitigated=0
            for idx, count in jobResultCounts.items():
                if count/maxCount < 0.5 or idx == "0x0":    # only filter counts which are less than half of the chance
                    # pretty complicated line, but we are converting just from hex indexing to binary here and padding zeros where necessary
                    # filterResultCounts[bin_zero_padded]: idx:hex -> bin -> bin zero padded 
                    # mitigatedResult.results[0].data.counts[idx] = max(0,count - self.filterResultCounts[format(int(idx,16), f'0{int(log2(nQubits))}b')])
                    if idx in self.filterResultCounts:
                        mitigatedResult.results[0].data.counts[idx] = max(0,count - self.filterResultCounts[idx])
                        nMitigated+=1
                    # it can (and often will) happen, that the result list contains keys which are not in the filter result counts
                    # especially in large circuits, this is the case, as there are so many computational basis states (2**nQubits)
                    # that it's very unlikely every state is covered by just an initialized circuit (like the filter)

            if not self.suppressPrint:
                print(f"Mitigated {nMitigated} in total")

            return mitigatedResult
        else:
            if self.measFitter == None:
                print("Need to initialize measurement fitter first")
                if nQubits == None:
                    print(f"For auto-initialization, you must provide the number of qubits")
                    return jobResult
                self.setupMeasurementFitter(nQubits=nQubits)

            # Get the filter object
            measFilter = self.measFitter.filter

            # Results with mitigation
            mitigatedResult = measFilter.apply(jobResult)
            # mitigatedCounts = mitigatedResult.get_counts(0)
            print(f"Filtering achieved at '0000': {mitigatedResult.get_counts()['0000']} vs before: {jobResult.get_counts()['0000']}")
        return mitigatedResult

    def mitiqNoiseFilter(self, jobResult, nQubits:int):
        return jobResult

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
            if rot <= self.minRotation:
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

    def dense(self, y_hat, D=3):
        if D==0:
            return y_hat
        D = int(D)
        y_hat_densed = np.zeros(int(y_hat.size/D))

        for i in range(y_hat_densed.size*D):
            # if i%D != 0:
            y_hat_densed[int(i/D)] += abs(y_hat[i])

        return y_hat_densed

    def processQFT(self, y:np.array):
        n_samples = y.size
        assert isPow2(n_samples)

        nQubits = int((log2(n_samples)/log2(2)))
        if not self.suppressPrint:
            print(f"Using {nQubits} Qubits to encode {n_samples} Samples")     

        if y.max() == 0.0:
            if self.fixZeroSignal:
                print(f"Warning: Signal's max value is zero and therefore amplitude initialization will fail. Setting signal to constant-one to continue")
                y = np.ones(n_samples)
            else:
                if not self.suppressPrint:
                    print(f"Zero Signal and fix should not be applied. Will return zero signal with expected length")
                y_hat = np.zeros(2**nQubits)
                return y_hat

        # Normalize ampl, which is required for squared sum of amps=1
        ampls = y / np.linalg.norm(y)
        q = QuantumRegister(nQubits,'q')

        if self.transpileOnce and not self.transpiled:
            self.transpiledQ = QuantumRegister(nQubits,'q')
            self.transpiledQC = QuantumCircuit(self.transpiledQ)
            self.transpiledQC = self.qft(self.transpiledQC, nQubits)
            self.transpiledQC.measure_all()
    

            if not self.suppressPrint:
                print(f"Transpiling for {self.backend}")
            if not self.suppressPrint:
                print(f"Depth before transpiling: {self.transpiledQC.depth()}")

            self.transpiledQC = transpile(self.transpiledQC, self.backend, optimization_level=self.transpOptLvl) # opt level 0,1..3. 3: heaviest opt

            if not self.suppressPrint:
                print(f"Depth after transpiling: {self.transpiledQC.depth()}")

            qc = QuantumCircuit(q, name="qft circuit")
            qc.initialize(ampls, [q[i] for i in range(nQubits)])
            qc = transpile(qc, self.backend, optimization_level=self.transpOptLvl)
            qc = qc + self.transpiledQC

            self.transpiled = True
        elif self.transpileOnce and self.transpiled:
            qc = QuantumCircuit(q, name="qft circuit")
            qc.initialize(ampls, [q[i] for i in range(nQubits)])
            qc = transpile(qc, self.backend, optimization_level=self.transpOptLvl)
            qc = qc + self.transpiledQC
        else:
            qc = QuantumCircuit(q, name="qft circuit")

            # for 2^n amplitudes, we have n qubits for initialization
            # this means that the binary representation happens exactly here
            qc.initialize(ampls, [q[i] for i in range(nQubits)])
            qc = self.qft(qc, nQubits)
            qc.measure_all()
            qc = transpile(qc, self.backend, optimization_level=self.transpOptLvl) # opt level 0,1..3. 3: heaviest opt

        if self.draw:
            self.draw=False
            name = str(time.mktime(datetime.datetime.now().timetuple()))[:-2]
            qc.draw(output='mpl', filename=f'./export/{name}.png')

        


        # qc = transpile(qc, self.backend, optimization_level=self.transpOptLvl) # opt level 0,1..3. 3: heaviest opt


        if not self.suppressPrint:
            print("Executing job...")
    
        #substitute with the desired backend
        job = execute(qc, self.backend,shots=self.numOfShots,noise_model=self.noiseModel)
        # if job.status != "COMPLETED":
        if not self.suppressPrint:
            job_monitor(job, interval=5) #run a blocking monitor thread

        # self.lastJobResultCounts = job.result().get_counts()
        
        if not self.suppressPrint:
            print("Post Processing...")
        
        if self.noiseMitigationOpt == 1:
            jobResult = self.qubitNoiseFilter(jobResult=job.result(), nQubits=nQubits)
        if self.noiseMitigationOpt == 2:
            jobResult = self.mitiqNoiseFilter(jobResult=job.result(), nQubits=nQubits)
        else:
            if not self.suppressPrint:
                print("Warning: Mitigating results is implicitly disabled. Consider enabling it by running 'setupMeasurementFitter'")
            jobResult = job.result()

        counts = jobResult.get_counts()
        y_hat = np.array(get_fft_from_counts(counts, nQubits))

        # [:n_samples//2]
        # y_hat = self.dense(y_hat, D=max(n_qubits/(self.samplingRate/n_samples),1))

        # Omitting normalization here, since we normalize in post
        y_hat = y_hat * 1/self.numOfShots
        # y_hat = y_hat*(1/y_hat.max())

        # top_indices = np.argsort(-np.array(fft))
        # freqs = top_indices*self.samplingRate/n_samples
        # get top 5 detected frequencies

        
        return y_hat

    # def executor(circuit: mitiq.QPROGRAM) -> float:
    #     pass

    def processIQFT(self, y:np.array):
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

