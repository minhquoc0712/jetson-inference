from torch import embedding
from utils.tensorrt import get_engine
from utils.onnx import pytorch_to_onnx

import pycuda.autoinit
import pycuda.driver as cuda
from opendr.engine.learners import Learner
import os
import numpy as np
from logging import getLogger

from typing import Union
from benchmark_class.base_class import baseClassBenchmark
import tensorrt as trt
import time

logger = getLogger(__name__)

class FRCBenchmark(baseClassBenchmark):
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
        super(FRCBenchmark, self).__init__(
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
        
        self.onnx_backbone_file = os.getcwd() + f'/{self.learner.backbone}.onnx'
        self.onnx_head_file = os.getcwd() + '/classifier.onnx'
        self.engine_backbone_file_path = self.onnx_backbone_file.replace('.onnx', '.trt')
        self.engine_head_file_path = self.onnx_head_file.replace('.onnx', '.trt')


    def download_weights(self):
        logger.info(
            "Use random intialized model for inference."
        )

    def optimize(
        self,
        input_names=['input'],
        output_names=['output'],
        iter=100
    ):
        logger.info(
            f"Export to Onnx model with enable_optimization is {self.enable_optimization}"
        )

        pytorch_to_onnx(
            pytorch_model=self.learner.backbone_model,
            filename=self.onnx_backbone_file,
            device=self.learner.device,
            provider=self.provider,
            input_shape=self.benchmark_input_shape,
            input_names=input_names,
            output_names=output_names,
            iter=iter,
            enable_optimization=self.enable_optimization,
            verification=False
        )

        if self.learner.network_head == 'classifier':

            logger.info(
                f'Export classifier head to {self.onnx_head_file}.'
            )
            
            pytorch_to_onnx(
                pytorch_model=self.learner.network_head_model,
                filename=self.onnx_head_file,
                device=self.learner.device,
                provider=self.provider,
                input_shape=[1, self.learner.embedding_size],
                input_names=input_names,
                output_names=output_names,
                iter=iter,
                enable_optimization=self.enable_optimization,
                verification=True
            )

    def allocate_memory(self, batch):
        self.d_input = None
        self.d_embedding_out = None
        self.stream = cuda.Stream()
       
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            print(size)
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.d_input = cuda.mem_alloc(1 * batch.nbytes)
            else:
                self.output_temp = np.empty(size, dtype=self.target_dtype)
                self.d_embedding_out = cuda.mem_alloc(1 * self.output_temp.nbytes)

        assert self.d_input is not None
        assert self.d_embedding_out is not None
        
        self.bindings_backbone = [int(self.d_input), int(self.d_embedding_out)]


        if self.learner.mode != 'backbone_only' and self.learner.network_head == 'classifier':
            self.stream_head = cuda.Stream()
            
            for binding in self.engine_head:
                size = trt.volume(self.engine_head.get_binding_shape(binding)) * self.engine_head.max_batch_size
                print(size)
                # Append to the appropriate list.
                if self.engine_head.binding_is_input(binding):
                    self.d_embedding_in = cuda.mem_alloc(1 * self.output_temp.nbytes)
                else:
                    self.output = np.empty(size, dtype=self.target_dtype)
                    self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
            
            assert self.d_embedding_in is not None
            assert self.d_output is not None

            self.bindings_head = [int(self.d_embedding_in), int(self.d_output)]

    def create_engine(self):
        self.engine = get_engine(
            onnx_file_path=self.onnx_backbone_file,
            precision=self.precision,
            strict_precision=self.strict_precision,
            engine_file_path=self.engine_backbone_file_path
        )
    
        self.context = self.engine.create_execution_context()

        if self.learner.mode != 'backbone_only' and self.learner.network_head == 'classifier':
            logger.info(
                'Create tensorrt engine for network head'
            )
            self.engine_head = get_engine(
                onnx_file_path=self.onnx_head_file,
                precision=self.precision,
                strict_precision=self.strict_precision,
                engine_file_path=self.engine_head_file_path
            )
    
            self.context_head = self.engine_head.create_execution_context()

        
    def infer(self, batch): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings_backbone, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output_temp, self.d_embedding_out, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        if self.learner.mode != 'backbone_only' and self.learner.network_head == 'classifier':
            # Transfer input data to device
            cuda.memcpy_htod_async(self.d_embedding_in, self.output_temp, self.stream_head)
            # Execute model
            self.context.execute_async_v2(self.bindings_head, self.stream_head.handle, None)
            # Transfer predictions back
            cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream_head)
            # Syncronize threads
            self.stream_head.synchronize()

            return self.output
        else:
            return self.output_temp

    def tensorrt_benchmark(self):
        def get_device_fn(*args):
            return next(self.learner.model.parameters()).device

        model_infer = self.infer
        get_device_fn = lambda x : None

        self.call_benchmark_module(model_infer, get_device_fn)
