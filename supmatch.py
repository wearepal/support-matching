from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option
from omegaconf import DictConfig, MISSING
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from src.arch.autoencoder import AeFromArtifact, ResNetAE, SimpleConvAE, VqGanAe
from src.arch.predictors.fcn import Fcn, SetFcn
from src.data.nih import NIHChestXRayDataset
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    LabelFromArtifact,
    NullLabeller,
    UniformLabelNoiser,
)
from src.models.discriminator import NeuralDiscriminator
from src.relay import SupMatchRelay

cs = ConfigStore.instance()
cs.store(node=SupMatchRelay, name="config_schema")
for package, entries in SupMatchRelay.options.items():
    for name, node in entries.items():
        cs.store(node=node, name=name, package=package, group=f"schema/{package}")


@hydra.main(config_path="external_confs", config_name="config", version_base="1.2")
def main(hydra_config: DictConfig) -> None:
    # ds_ops = [
    #     Option(ColoredMNIST, name="cmnist"),
    #     Option(CelebA, name="celeba"),
    #     Option(Camelyon17, name="camelyon17"),
    #     Option(NIHChestXRayDataset, name="nih"),
    # ]
    # ae_arch_ops = [
    #     Option(AeFromArtifact, name="artifact"),
    #     Option(ResNetAE, name="resnet"),
    #     Option(SimpleConvAE, name="simple"),
    #     Option(VqGanAe, name="vqgan"),
    # ]
    # disc_arch_ops = [
    #     Option(Fcn, name="sample"),
    #     Option(SetFcn, name="set"),
    # ]
    # labeller_ops = [
    #     Option(CentroidalLabelNoiser, name="centroidal_noise"),
    #     Option(GroundTruthLabeller, name="gt"),
    #     Option(KmeansOnClipEncodings, name="kmeans"),
    #     Option(LabelFromArtifact, name="artifact"),
    #     Option(NullLabeller, name="none"),
    #     Option(UniformLabelNoiser, name="uniform_noise"),
    # ]
    omega_dict = instantiate(hydra_config)
    exp = OmegaConf.to_object(omega_dict)
    assert isinstance(exp, SupMatchRelay)
    exp.run(
        OmegaConf.to_container(hydra_config, throw_on_missing=True, enum_to_str=False, resolve=True
    ))

    # SupMatchRelay.with_hydra(
    #     ae_arch=ae_arch_ops,
    #     disc=[Option(NeuralDiscriminator, name="base")],
    #     disc_arch=disc_arch_ops,
    #     ds=ds_ops,
    #     labeller=labeller_ops,
    #     instantiate_recursively=False,
    #     clear_cache=True,
    #     root="conf",
    # )


if __name__ == "__main__":
    main()
