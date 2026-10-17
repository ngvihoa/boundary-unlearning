"""Dataset and attack normalization used by the experiment."""


CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CIFAR_CLASSES)}
CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD = (0.5, 0.5, 0.5)
REPO_ATTACK_MEAN = (0.4914, 0.4822, 0.2265)
REPO_ATTACK_STD = (0.2023, 0.1994, 0.2010)
