benchmark folder was copied from: https://github.com/LukasHedegaard/pytorch-benchmark. Some modifications was added.
Code for running inference on onnx and tensorrt was copied from Dat Thanh Tran scripts with some modifications.

## Benchmark Workflow:

1. Install the dependencies needed for running a learner.
2. Create benchmark_class config and model config files.
3. Create learner wrapper class (optional) and benchmark class.
4. Change hydra output working directory for descriptive name.
5. Implement and run pytorch inference.
6. Implement and run onnx inference.
7. Implement and run tensorrt inference.

Example:

```
python main.py benchmark_class=pstbln benchmark_class/model=pstbln_casia mode=tensorrt
```