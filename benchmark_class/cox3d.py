import pycuda.autoinit
import pycuda.driver as cuda
import os
from pathlib import Path
from logging import getLogger
import os

from hydra.utils import instantiate
from pathlib import Path
from logging import getLogger

from typing import Union
from opendr.engine.learners import Learner

from benchmark.onnx_benchmark import pytorch_to_onnx

logger = getLogger(__name__)

class CoX3DBenchmark():
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
        super(CoX3DBenchmark, self).__init__(
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
            f"Load model weights from {self.pytorch_model_dir}/x3d_{self.learner.backbone}.pyth"
        )   
        self.learner.load(Path(self.pytorch_model_dir) / f"x3d_{self.learner.backbone}.pyth")  # x3d

    
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
            enable_optimization=self.enable_optimization,
            verification=False
        )

