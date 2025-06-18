import hydra


@hydra.main(config_path="../../configs", config_name="hpo_crl", version_base="1.1")
def hpo_target(cfg):
    print(cfg)
    return 10


if __name__ == "__main__":
    hpo_target()
