from benchmark.onnx_benchmark import pytorch_to_onnx, OnnxModel
import os
import torch
import onnxruntime as ort
from opendr.perception.face_recognition.algorithm.util.utils import l2_norm
import numpy as np

def optimize_frc(
        learner, 
        benchmark_input_shape, 
        provider,
        input_names,
        output_names,
        iter,
        enable_optimization
    ):

    path_backbone = os.getcwd() + f'/{learner.backbone}.onnx'

    pytorch_to_onnx(
        pytorch_model=learner.backbone_model,
        filename=path_backbone,
        device=learner.device,
        provider=provider,
        input_shape=benchmark_input_shape,
        input_names=input_names,
        output_names=output_names,
        iter=iter,
        enable_optimization=enable_optimization
    )

    if learner.network_head == 'classifier':
        print('----------------Hello-----------------')
        path_network_head = os.getcwd() + f'/{learner.network_head}.onnx'
        pytorch_to_onnx(
            pytorch_model=learner.network_head_model,
            filename=path_network_head,
            device=learner.device,
            provider=provider,
            input_shape=[1, learner.embedding_size],
            input_names=input_names,
            output_names=output_names,
            iter=iter,
            enable_optimization=enable_optimization
        )


def benchmark_fcr_classifier_head(learner, mode, provider):
    print(f'------------------backbone: {learner.backbone} ------------network_head: {learner.network_head}----------')
    if mode == 'onnx':
        path_backbone = os.getcwd() + f'/{learner.backbone}.onnx'
        path_network_head = os.getcwd() + f'/{learner.network_head}.onnx'
        model_infer = OnnxFRC(
            engine_file_backbone=path_backbone,
            engine_file_head=path_network_head,
            learner=learner,
            provider=provider
        )
        get_device_fn = lambda x : torch.device('cuda')
        print('----------------Hello-----------------')
    elif mode == 'torch':
        model_infer = learner.direct_infer_in_torch
        get_device_fn = lambda x : torch.device('cuda')
    else:
        exit()

    return model_infer, get_device_fn


class OnnxFRC():
    def __init__(
        self, 
        engine_file_backbone, 
        engine_file_head,
        learner,
        provider='CPUExecutionProvider',
    ):

        print('all available execution providers')
        print('---')
        providers = ort.get_available_providers()
        for p in providers:
            print(p)
        print('---')
        print(f'trying to run with {provider}')
        print('---')

        self.network_head = learner.network_head
        self.mode = learner.mode

        self.ort_backbone_session = ort.InferenceSession(
            engine_file_backbone,
            providers=[provider]
        )
        self.input_name_backbone = self.ort_backbone_session.get_inputs()[0].name
        print(self.ort_backbone_session.get_providers())

        if self.network_head == 'classifier':
            self.ort_head_session = ort.InferenceSession(
                engine_file_head,
                providers=[provider]
            )
            self.input_name_head = self.ort_head_session.get_inputs()[0].name
            print(self.ort_head_session.get_providers())


    def __call__(self, inputs):
        if self.mode == 'backbone_only':
            with torch.no_grad():
                features = self.ort_backbone_session.run(None, {self.input_name_backbone: inputs})
                features = torch.tensor(features[0])
                features = l2_norm(features)
                return features
        elif self.network_head == 'classifier':
            with torch.no_grad():
                features = self.ort_backbone_session.run(None, {self.input_name_backbone: inputs})

                features = torch.tensor(features[0])
                features = l2_norm(features)

                outs = self.ort_head_session.run(None, {self.input_name_head: np.array(features)})
                return outs
        else:
            raise RuntimeError

