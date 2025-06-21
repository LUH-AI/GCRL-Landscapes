import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="hpo_crl", version_base="1.1")
def hpo_target(cfg: DictConfig) -> float:
    print(cfg)
    return 10


if __name__ == "__main__":
    hpo_target()
