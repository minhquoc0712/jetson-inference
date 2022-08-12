from torch import embedding
from utils.tensorrt import get_engine
from utils.onnx import pytorch_to_onnx, OnnxModel

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
from benchmark import benchmark
import torch
import onnxruntime as ort
import tempfile


logger = getLogger(__name__)

class LightweightOpenPoseBenchmark(baseClassBenchmark):
    def __init__(
        self,
        model: Learner,
        name: str,
        example_input_shape: Union[list, tuple],
        mode: str,
        pytorch_model_dir: str,
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
        super(LightweightOpenPoseBenchmark, self).__init__(
            model=model,
            name=name,
            example_input_shape=example_input_shape,
            mode=mode,
            pytorch_model_dir=pytorch_model_dir,
            provider=provider,
            batch_size=batch_size,
            warmup_iter=warmup_iter,
            benchmark_iter=benchmark_iter,
            enable_optimization=enable_optimization,
            precision=precision,
            strict_precision=strict_precision,
            num_class=num_class
        )

        if self.learner.half:
            self.precision = 'FP16'

        self.onnx_file_path = os.getcwd() + f'/{self.learner.backbone}.onnx'
        self.engine_file_path = self.onnx_file_path.replace('.onnx', 'trt')

    def download_weights(self):
        logger.info(
            f'Download pretrained model to {self.pytorch_model_dir}'
        )
        self.learner.download(path=self.pytorch_model_dir, mode='pretrained', verbose=True)


    def load_weights(self):
        logger.info(
            f'Load pretrained model from {self.pytorch_model_dir}/openpose_default'
        )
        self.learner.load(self.pytorch_model_dir + "/openpose_default")


    def optimize(self):
        if not os.path.exists(self.onnx_file_path):
            # turn pytorch model into eval mode
            self.learner.model.eval()

            if self.learner.num_refinement_stages == 2:
                output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                                'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']
                target_np_type = 'float32'
                target_torch_type = torch.float32
                atol = 1e-3
            else:
                output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs']
                target_np_type = 'float16'
                target_torch_type = torch.float16
                atol = 1e-2

            logger.info(f'onnx models output names: {output_names}')

            # conversion
            if self.enable_optimization:
                # create tmp file
                tmp_path = os.path.join(tempfile.gettempdir(), '{}.onnx'.format(time.time()))
                torch.onnx.export(
                    self.learner.model,
                    torch.randn(*self.benchmark_input_shape, dtype=target_torch_type).to(self.learner.device),
                    tmp_path,
                    do_constant_folding=True,
                    export_params=True, # if set to False exports untrained model
                    input_names=['input'],
                    output_names=output_names,
                    opset_version=11,
                )

                sess_options = ort.SessionOptions()
                sess_options.enable_profiling = True
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.optimized_model_filepath = self.onnx_file_path

                session = ort.InferenceSession(tmp_path, sess_options=sess_options, providers=[self.provider])
                input_name = session.get_inputs()[0].name
                for i in range(100):
                    inputs = np.random.rand(*self.benchmark_input_shape).astype(target_np_type)
                    session.run([], {input_name: inputs})

                assert os.path.exists(self.onnx_file_path)
            else:
                torch.onnx.export(
                    self.learner.model,
                    torch.randn(*self.benchmark_input_shape, dtype=target_torch_type).to(self.learner.device),
                    self.onnx_file_path,
                    do_constant_folding=True,
                    export_params=True,
                    input_names=['input'],
                    output_names=output_names,
                    opset_version=11
                )

            # verification
            onnx_model = OnnxModel(self.onnx_file_path, provider=self.provider)
            for i in range(10):
                x = torch.randn(*self.benchmark_input_shape, dtype=target_torch_type)
                with torch.no_grad():
                    out_torch = self.learner.model(x.to(self.learner.device))

                out_onnx = onnx_model(x.numpy())

                assert len(out_onnx) == len(out_torch)
                out_onnx = [out_onnx[i].flatten() for i in range(len(out_onnx))]
                out_torch = [out_torch[i].cpu().numpy().flatten() for i in range(len(out_torch))]

                for j in range(len(out_onnx)):
                    if not np.allclose(out_onnx[j], out_torch[j], atol=atol):
                        print('mismatched prediction outputs')
                        print(f'--onnx output: {out_onnx[j]}')
                        print(f'--pytorch output: {out_torch[j]}')
                        raise RuntimeError()
            print('complete verification')
        else:
            logger.info("Load existing onnx file")

    def pytorch_benchmark(self):
        def get_device_fn(*args):
            return next(self.learner.model.parameters()).device
        
        model_infer = self.learner.model
        
        self.call_benchmark_module(model_infer, get_device_fn)


    def allocate_memory(self, batch):
        self.d_input = None
        self.d_output = []
        self.bindings = []
        self.stream = cuda.Stream()

        i = 0
        j = 0
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.d_input = HostDeviceMem(host_mem, device_mem)
            else:
                self.d_output.append(HostDeviceMem(host_mem, device_mem))

        assert self.d_input is not None
        assert self.d_output is not None


    def tensorrt_infer(self, batch):
        if self.stream is None:
            self.allocate_memory(batch)
        
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.d_input.device, batch, self.stream)
        
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.d_output]
        
        # Synchronize the stream
        self.stream.synchronize()
        
        # Return only the host outputs.
        return [out.host for out in self.d_output]


    def call_benchmark_module(self, model_infer, get_device_fn):
        if self.mode == 'torch':
            def transfer_device_fn(sample, device):
                if isinstance(sample, list):
                    return [sample[i].to(device) for i in range(len(sample))]
                elif isinstance(sample, torch.Tensor):
                    return sample.to(device)
                else:
                    raise RuntimeError
        else:
            transfer_device_fn = lambda x : None


        if self.learner.half:
            def generate_input_sample(input_shape: Union[list, tuple], mode) -> Union[torch.Tensor, np.array]:
                if mode == 'torch':
                    return torch.randn(*input_shape, dtype=torch.float16).to('cpu')
                else:
                    return np.random.rand(*input_shape).astype('float16')
        else:
            def generate_input_sample(input_shape: Union[list, tuple], mode) -> Union[torch.Tensor, np.array]:
                if mode == 'torch':
                    return torch.randn(*input_shape, dtype=torch.float32).to('cpu')
                else:
                    return np.random.rand(*input_shape).astype('float32')

        benchmark(
            model=model_infer,
            input_shape=self.benchmark_input_shape,
            num_runs=self.benchmark_iter,
            warm_up_iter=self.warmup_iter,
            mode=self.mode,
            print_fn=print,
            transfer_to_device_fn=transfer_device_fn,
            generate_input_sample_fn=generate_input_sample,
            get_device_fn=get_device_fn
        )


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()