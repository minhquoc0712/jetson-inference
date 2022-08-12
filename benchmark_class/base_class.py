from utils.tensorrt import get_engine
from utils.onnx import pytorch_to_onnx, OnnxModel

import pycuda.autoinit
import pycuda.driver as cuda
from opendr.engine.learners import Learner
import os
from benchmark.benchmark import benchmark
from logging import getLogger

from pathlib import Path
import numpy as np
from logging import getLogger

from typing import Union
from benchmark.benchmark import benchmark
import numpy as np
import torch

logger = getLogger(__name__)

class baseClassBenchmark():
    def __init__(
        self,
        model: Learner,
        name: str,
        example_input_shape: Union[list, tuple],
        mode: str,
        pytorch_model_dir: str,
        provider: str,
        batch_size: int = 1,
        warmup_iter: int = 10,
        benchmark_iter: int = 100,
        enable_optimization: bool = False,
        precision: str = 'FP32',
        strict_precision: bool = False,
        num_class: bool = 0,
        *args,
        **kwargs
    ):
        self.name = name
        self.learner = model
        self.model_device = self.learner.device
        self.batch_size = batch_size
        self.warmup_iter = warmup_iter
        self.benchmark_iter = benchmark_iter
        self.pytorch_model_dir = pytorch_model_dir
        self.provider = provider
        self.enable_optimization = enable_optimization

        self.precision = precision
        self.strict_precision = strict_precision

        self.num_class = num_class

        if self.precision == 'FP32':
            self.target_dtype = np.float32
        elif self.precision == 'FP16':
            print('-------------Use FP16---------------')
            self.target_dtype = np.float16
        else:
            raise RuntimeError("Unknown precision.")

        self.stream = None

        assert mode in ['torch', 'onnx', 'tensorrt'], \
        'Wrong benchmark mode. Choose one in \["torch", "onnx", "tensorrt"\]'
        self.mode = mode

        if hasattr(self.learner, '_example_input'):
            assert list(example_input_shape) == list(getattr(self.learner, '_example_input').shape), \
            f"{list(example_input_shape)} difference with {list(getattr(self.learner, '_example_input').shape)}"

        self.benchmark_input_shape = (self.batch_size, *example_input_shape[1:])
        logger.info(
            f"Shape of benchmark sample: {self.benchmark_input_shape}."
        )
     
    def download_weights(self):
        return

    def load_weights(self):
        return

    
    def optimize(
        self,
        input_names=['input'],
        output_names=['output'],
        iter=100
    ):
        logger.info(
            f"Export to Onnx model with enable_optimize is {self.enable_optimization}"
        )
        if not os.path.exists(self.onnx_file_path):
            pytorch_to_onnx(
                pytorch_model=self.learner.model, 
                filename=self.onnx_file_path,
                device=self.learner.device,
                provider=self.provider,
                input_shape=self.benchmark_input_shape, 
                input_names=input_names,
                output_names=output_names,
                iter=iter,
                enable_optimization=self.enable_optimization,
                do_constant_folding=True,
                verification=True
            )
        else:
            logger.info("Load existing onnx file")


    def create_engine(self):
        self.engine_file_path = self.onnx_file_path.replace('.onnx', '.trt')
        self.engine = get_engine(
            onnx_file_path=self.onnx_file_path,
            precision=self.precision,
            strict_precision=self.strict_precision,
            engine_file_path=self.engine_file_path
        )
    
        self.context = self.engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(self.num_class, dtype=self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16
        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()


    def tensorrt_infer(self, batch): # result gets copied into output
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
        def get_device_fn(*args):
            return next(self.learner.model.parameters()).device

        model_infer = self.tensorrt_infer
        get_device_fn = lambda x : None

        self.call_benchmark_module(model_infer, get_device_fn)

    def onnx_benchmark(self):
        model_infer = OnnxModel(
            engine_file=self.onnx_file_path,
            provider=self.provider
        )

        get_device_fn = lambda x : None

        self.call_benchmark_module(model_infer, get_device_fn)

    def pytorch_benchmark(self):
        def get_device_fn(*args):
            return next(self.learner.model.parameters()).device
        
        model_infer = self.learner.model
        
        self.call_benchmark_module(model_infer, get_device_fn)

    def call_benchmark_module(self, model_infer, get_device_fn):
        benchmark(
            model=model_infer,
            input_shape=self.benchmark_input_shape,
            num_runs=self.benchmark_iter,
            warm_up_iter=self.warmup_iter,
            mode=self.mode,
            print_fn=print,
            get_device_fn=get_device_fn
        )

