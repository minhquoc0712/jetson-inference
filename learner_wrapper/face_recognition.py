import torch
import numpy as np
from opendr.perception.face_recognition.algorithm.util.utils import l2_norm
from opendr.perception.face_recognition import FaceRecognitionLearner

from opendr.perception.face_recognition.algorithm.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from opendr.perception.face_recognition.algorithm.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, \
    IR_SE_152
from opendr.perception.face_recognition.algorithm.backbone.model_mobilenet import MobileFaceNet
from opendr.perception.face_recognition.algorithm.head.losses import ArcFace, CosFace, SphereFace, AMSoftmax, Classifier


class FaceRecognitionWrapper(FaceRecognitionLearner):
    def __init__(self, lr=0.1, iters=120, batch_size=128, optimizer='sgd', device='cuda', threshold=0.0,
                 backbone='ir_50', network_head='arcface', loss='focal',
                 temp_path='./temp', mode='backbone_only',
                 checkpoint_after_iter=0, checkpoint_load_iter=0, val_after=0,
                 input_size=[112, 112], rgb_mean=[0.5, 0.5, 0.5], rgb_std=[0.5, 0.5, 0.5], embedding_size=512,
                 weight_decay=5e-4, momentum=0.9, drop_last=True, stages=[35, 65, 95],
                 pin_memory=True, num_workers=4, num_class=0,
                 seed=123,
                 *args,
                 **kwargs):
        super(FaceRecognitionWrapper, self).__init__(
            lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer, device=device, threshold=threshold,
            backbone=backbone, network_head=network_head, loss=loss,
            temp_path=temp_path, mode=mode,
            checkpoint_after_iter=checkpoint_after_iter, checkpoint_load_iter=checkpoint_load_iter, val_after=val_after,
            input_size=input_size, rgb_mean=rgb_mean, rgb_std=rgb_std, embedding_size=embedding_size,
            weight_decay=weight_decay, momentum=momentum, drop_last=drop_last, stages=stages,
            pin_memory=pin_memory, num_workers=num_workers,
            seed=seed
        )
        
        if self.mode == 'backbone_only':
            self.num_class = 0
            print(f'--------------Only use backbone. num_class: {self.num_class}--------------')
        elif self.network_head == 'classifier':
            self.num_class = num_class
            print(f'-----------Create network head model. num_class: {self.num_class}----------------')
        else:
            raise RuntimeError
            
        self.init_model(self.num_class)

    def init_model(self, num_class):
        # Create the backbone architecture
        self.num_class = num_class
        if self.backbone_model is None:
            backbone_dict = {'resnet_50': ResNet_50(self.input_size),
                             'resnet_101': ResNet_101(self.input_size),
                             'resnet_152': ResNet_152(self.input_size),
                             'ir_50': IR_50(self.input_size),
                             'ir_101': IR_101(self.input_size),
                             'ir_152': IR_152(self.input_size),
                             'ir_se_50': IR_SE_50(self.input_size),
                             'ir_se_101': IR_SE_101(self.input_size),
                             'ir_se_152': IR_SE_152(self.input_size),
                             'mobilefacenet': MobileFaceNet()}
            backbone = backbone_dict[self.backbone]
            self.backbone_model = backbone.to(self.device)
        # Create the head architecture
        if self.mode != 'backbone_only':
            head_dict = {
                'arcface': ArcFace(in_features=self.embedding_size, out_features=self.num_class, device=self.device),
                'cosface': CosFace(in_features=self.embedding_size, out_features=self.num_class, device=self.device),
                'sphereface': SphereFace(in_features=self.embedding_size, out_features=self.num_class,
                                         device=self.device),
                'am_softmax': AMSoftmax(in_features=self.embedding_size, out_features=self.num_class,
                                        device=self.device),
                'classifier': Classifier(in_features=self.embedding_size, out_features=self.num_class,
                                         device=self.device)}
            head = head_dict[self.network_head]
            self.network_head_model = head.to(self.device)
        else:
            self.network_head_model = None

    def direct_infer_in_torch(self, sample):
        if self.mode == 'backbone_only':
            self.backbone_model.eval()
            with torch.no_grad():
                self.backbone_model.eval()
                features = self.backbone_model(sample)
                outs = l2_norm(features)
                return outs
        elif self.network_head == 'classifier':
            self.backbone_model.eval()
            self.network_head_model.eval()
            with torch.no_grad():
                self.backbone_model.eval()
                features = self.backbone_model(sample)
                features = l2_norm(features)
                outs = self.network_head_model(features)
                return outs
        else:
            raise RuntimeError

    def direct_infer_in_onnx(self, sample):
        if self.mode == 'backbone_only':
            self.backbone_model.eval()
            with torch.no_grad():
                features = self.ort_backbone_session.run(None, {'data': np.array(sample.cpu())})
                features = torch.tensor(features[0])
                features = l2_norm(features)
        elif self.network_head == 'classifier':
            self.backbone_model.eval()
            self.network_head_model.eval()
            with torch.no_grad():
                features = self.ort_backbone_session.run(None, {'data': np.array(sample.cpu())})

                features = torch.tensor(features[0])
                features = l2_norm(features)

                outs = self.ort_head_session.run(None, {'features': np.array(features.cpu())})
                return self.classes[outs.index(max(outs))]
        else:
            raise UserWarning('Infer should be called either with backbone_only mode or with a classifier head')
