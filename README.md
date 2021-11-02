# Hybrid Quantum Speech Processing - STQFT repository

This is the stqft directory of the *hqsp* project

## Usage

Experiments are indicated by [sc_*.py](sc_*.py) and don't require further arguments.

The viewer for the [data sub-repository](https://github.com/stroblme/hqsp-stqft-data) can be started with [Viewer.py](Viewer.py).
A specific experiment can be selected in subsequent command line dialogs.

## Structure

Time invariant transformations: [dft.py](dft.py), [fft.py](fft.py), [qft.py](qft.py).
Short time transformations: [stt.py](stt.py).
Specific time variant transformations: [stft.py](stft.py), [stqft.py](stqft.py)

Pre- and postprocessing, general signal structure and plotting: [frontend.py](frontend.py).
Threshold filter and Pow2 check: [utils.py](utils.py).
