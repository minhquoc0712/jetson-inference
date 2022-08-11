from tensorrt_wrapper.build_engine import get_engine
from tensorrt_wrapper.common import allocate_buffers
import pycuda.autoinit
import pycuda.driver as cuda
from opendr.engine.learners import Learner
import os
from benchmark.onnx_benchmark import pytorch_to_onnx, OnnxModel
from pathlib import Path
from benchmark.benchmark import benchmark
from logging import getLogger
import os

from hydra.utils import instantiate
from pathlib import Path
import onnxruntime as ort
import numpy as np
from logging import getLogger

from typing import Union
from opendr.engine.learners import Learner

from benchmark.onnx_benchmark import pytorch_to_onnx, OnnxModel
from benchmark.benchmark import benchmark
import numpy as np

logger = getLogger(__name__)

class X3DBenchmark():
    def __init__(
        self,
        model: Learner,
        name: str,
        pytorch_model_dir: str,
        example_input_shape: Union[list, tuple],
        mode: str,
        provider: str,
        num_class: bool,
        batch_size: int = 1,
        warmup_iter: int = 10,
        benchmark_iter: int = 100,
        enable_optimization: bool = False,
        precision: str = 'FP32',
        strict_precision: bool = False,
        *args,
        **kwargs
    ):
        super(X3DBenchmark, self).__init__(
            model=model,
            name=name,
            pytorch_model_dir=pytorch_model_dir,
            example_input_shape=example_input_shape,
            mode=mode,
            provider=provider,
            batch_size=batch_size,
            warmup_iter=warmup_iter,
            benchmark_iter=benchmark_iter,
            enable_optimization=enable_optimization,
            precision=precision,
            strict_precision=strict_precision,
            num_class=num_class
        )
     
    def download_weights(self):
        self.learner.download(path=self.pytorch_model_dir, model_names={self.learner.backbone})  # x3d

    def load_weights(self):
        logger.info(
            f"Load model weights from {self.pytorch_model_dir}/{self.name}.pyth"
        )   
        self.learner.load(Path(self.pytorch_model_dir) / f"{self.name}.pyth")  # x3d

    
    def optimize(
        self,
        input_names=['input'],
        output_names=['output'],
        iter=100
    ):
        logger.info(
            f"Export to Onnx model with enable_optimize is {self.enable_optimization}"
        )
        self.onnx_file_path = os.getcwd() + f'/{self.name}.onnx'
        pytorch_to_onnx(
            pytorch_model=self.learner.model, 
            filename=self.onnx_file_path,
            device=self.learner.device,
            provider=self.provider,
            input_shape=self.benchmark_input_shape, 
            input_names=input_names,
            output_names=output_names,
            iter=iter,
            enable_optimization=self.enable_optimization
        )


    def create_engine(self):
        self.engine_file_path = self.onnx_file_path.replace('.onnx', '.trt')
        engine = get_engine(
            onnx_file_path=self.onnx_file_path,
            precision=self.precision,
            strict_precision=self.strict_precision,
            engine_file_path=self.engine_file_path
        )
    
        self.context = engine.create_execution_context()
        
    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype=self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16
        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()


    def infer(self, batch): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return self.output

    def tensorrt_benchmark(self):
        model_infer = self.infer
        get_device_fn = lambda x : None

        self.call_benchmark_module(model_infer, get_device_fn)


