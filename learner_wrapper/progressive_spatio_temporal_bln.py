from opendr.perception.facial_expression_recognition import ProgressiveSpatioTemporalBLNLearner
from opendr.engine.learners import Learner


class ProgressiveSpatioTemporalBLNWrapper(ProgressiveSpatioTemporalBLNLearner):
    def __init__(self, lr=1e-1, batch_size=128, optimizer_name='sgd', lr_schedule='',
                 checkpoint_after_iter=0, checkpoint_load_iter=0, temp_path='temp',
                 device='cuda', num_workers=32, epochs=400, experiment_name='pstbln_casia',
                 device_indices=[0], val_batch_size=128, drop_after_epoch=[400],
                 start_epoch=0, dataset_name='CASIA', num_class=6, num_point=309, num_person=1, in_channels=2,
                 blocksize=5, num_blocks=100, num_layers=10, topology=[],
                 layer_threshold=1e-4, block_threshold=1e-4,
                 *args,
                 **kwargs):
        # super(ProgressiveSpatioTemporalBLNWrapper, self).__init__(
        #     lr=lr, batch_size=batch_size, optimizer_name=optimizer_name, lr_schedule=lr_schedule,
        #     checkpoint_after_iter=checkpoint_after_iter, checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
        #     device=device, num_workers=num_workers, epochs=epochs, experiment_name=experiment_name,
        #     device_indices=device_indices, val_batch_size=val_batch_size, drop_after_epoch=drop_after_epoch,
        #     start_epoch=start_epoch, dataset_name=dataset_name, num_class=num_class, num_class=num_class, num_person=num_person, in_channels=in_channels,
        #     blocksize=blocksize, num_blocks=num_blocks, num_layers=num_layers, topology=topology,
        #     layer_threshold=layer_threshold, block_threshold=block_threshold
        # )

        super(ProgressiveSpatioTemporalBLNWrapper, self).__init__(
            device=device, dataset_name=dataset_name,
            num_class=num_class,
            num_point=num_point, num_person=1, in_channels=2,
            blocksize=5, topology=[15, 10, 15, 5, 5, 10]
        )

        self.init_model()

        