from torchvision.transforms import transforms, ToTensor, Normalize, CenterCrop


class GlomeruliTestConfig:
    batch_size = 64
    num_classes = 2
    aug = False
    train = False
    transform = transforms.Compose([
        CenterCrop(224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])

    def __init__(self):
        pass


class GlomeruliTestConfig2(GlomeruliTestConfig):
    num_classes = 2


class GlomeruliTestConfig12(GlomeruliTestConfig):
    num_classes = 12
