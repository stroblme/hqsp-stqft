from qft import qft_framework
from stt import stt_framework

class stqft_framework():
    def __init__(self, **kwargs):
        self.stt_inst = stt_framework(qft_framework, **kwargs)

    def transform(self, y_signal, **kwargs):
        return self.stt_inst.stt_transform(y_signal, **kwargs)

    def transformInv(self, y_signal, **kwargs):
        return self.stt_inst.stt_transformInv(y_signal, **kwargs)

    def postProcess(self, y_hat, f, t):
        return self.stt_inst.postProcess(y_hat, f, t)