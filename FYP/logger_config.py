import logging
import pynvml

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    def emit(self, record):
        try:
            msg = self.format(record)
            print(msg)
        except Exception:
            self.handleError(record)

logging.basicConfig(level=logging.INFO, handlers=[TqdmLoggingHandler()])
logger = logging.getLogger(__name__)

# Optional: GPU info
try:
    pynvml.nvmlInit()
    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    gpu_device = None
