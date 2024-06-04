import hydra
import torch

torch.set_float32_matmul_precision('high')

@hydra.main(config_path='config', config_name='main', version_base='1.3')
def main(cfg):
    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()

