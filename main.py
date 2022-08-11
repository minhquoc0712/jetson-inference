import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg : DictConfig) -> None:
    model_name = OmegaConf.to_container(HydraConfig.get().runtime.choices)['benchmark_class']

    assert cfg.mode in ['torch', 'onnx', 'tensorrt']
    enable_optimization = True
    if cfg.mode == 'tensorrt':
        enable_optimization = False

    print(OmegaConf.to_yaml(cfg))
    benchmark_class = instantiate(
        cfg.benchmark_class,
        mode=cfg.mode,
        pytorch_model_dir=cfg.pytorch_model_dir,
        name=model_name,
        example_input_shape=cfg.benchmark_class.model.example_input_shape,
        num_class=cfg.benchmark_class.model.num_class,
        enable_optimization=enable_optimization
    )

    if cfg.benchmark_class.download:
        benchmark_class.download_weights()
    
    if cfg.benchmark_class.load:
        benchmark_class.load_weights()

    if cfg.mode != 'torch':
        benchmark_class.optimize()
        
    if cfg.mode == 'torch':
        benchmark_class.pytorch_benchmark()
    elif cfg.mode == 'onnx':
        benchmark_class.onnx_benchmark()
    else:
        benchmark_class.create_engine()
        benchmark_class.tensorrt_benchmark()

if __name__ == "__main__":
    my_app()