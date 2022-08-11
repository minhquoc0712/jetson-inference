"""
ONNX related utility

* Copyright: 2022 Dat Tran
* Authors  : Dat Thanh Tran
* Date     : 05/2022
"""

import onnxruntime as rt
import tempfile
import time
import os
import numpy as np
from typing import Union


def pytorch_to_onnx(
    pytorch_model,
    filename: str,
    device: str,
    provider: str,
    input_shape: Union[list, tuple],
    input_names: list = ['input'],
    output_names: list = ['output'],
    iter: int = 20,
    do_constant_folding: bool = True,
    enable_optimization: bool = False,
    verification: bool = True
):
    """
    convert pytorch to onnx model & optimize on the hardware that is used to run
    the conversion
    """

    import torch

    # turn pytorch model into eval mode
    pytorch_model.eval()

    # conversion
    if enable_optimization:
        # create tmp file
        tmp_path = os.path.join(tempfile.gettempdir(), '{}.onnx'.format(time.time()))
        torch.onnx.export(
            pytorch_model,
            torch.randn(*input_shape).float().to(device),
            tmp_path,
            do_constant_folding=do_constant_folding,
            export_params=True, # if set to False exports untrained model
            input_names=input_names,
            output_names=output_names,
            opset_version=11,
        )

        sess_options = rt.SessionOptions()
        sess_options.enable_profiling = True
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = filename

        session = rt.InferenceSession(tmp_path, sess_options=sess_options, providers=[provider])
        input_name = session.get_inputs()[0].name
        for i in range(iter):
            inputs = np.random.rand(*input_shape).astype('float32')
            session.run([], {input_name: inputs})

        assert os.path.exists(filename)
    else:
        torch.onnx.export(
            pytorch_model,
            torch.randn(*input_shape).to(device),
            filename,
            do_constant_folding=do_constant_folding,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=11
        )

    if verification:
         # verification
        onnx_model = OnnxModel(filename, provider=provider)
        for i in range(10):
            x = np.random.rand(*input_shape).astype('float32')
            out_onnx = onnx_model(x)[0]
            with torch.no_grad():
                out_torch = pytorch_model(torch.from_numpy(x).to(device))
            out_onnx = out_onnx.flatten()
            out_torch = out_torch.cpu().numpy().flatten()
            if not np.allclose(out_onnx, out_torch, atol=1e-3):
                print('mismatched prediction outputs')
                print(f'--onnx output: {out_onnx}')
                print(f'--pytorch output: {out_torch}')
                raise RuntimeError()
        print('complete verification')

def optimize_onnx_model(
    input_path,
    output_path,
    input_shape,
    iterations=20,
):
    """
    optimize onnx model on the hardware that is used to run this script
    """

    sess_options = rt.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path
    session = rt.InferenceSession(input_path, sess_options=sess_options)
    input_name = session.get_inputs()[0].name
    for i in range(iter):
        inputs = np.random.rand(*input_shape).astype('float32')
        session.run([], {input_name: inputs})
    assert os.path.exists(output_path)

class OnnxModel:
    def __init__(self, engine_file, provider='CPUExecutionProvider'):
        print('all available execution providers')
        print('---')
        providers = rt.get_available_providers()
        for p in providers:
            print(p)
        print('---')
        print(f'trying to run with {provider}')
        print('---')
        self.session = rt.InferenceSession(
            engine_file,
            providers=[provider]
        )
        self.input_name = self.session.get_inputs()[0].name
        print(self.session.get_providers())

    def __call__(self, inputs):
        return self.session.run([], {self.input_name: inputs})
