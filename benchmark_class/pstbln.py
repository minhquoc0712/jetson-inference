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

class ProgressiveSpatioTemporalBLNBenchmark(baseClassBenchmark):
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
        super(ProgressiveSpatioTemporalBLNBenchmark, self).__init__(
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
                do_constant_folding=False,
                verification=True
            )