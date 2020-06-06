import torch
import torch.nn.functional as F


def cross_entropy(input, target):
    return F.cross_entropy(input=input, target=target)


def fetch_loss(name="cross_entropy"):
    print("Using loss: '%s'." % name)
    return LOSSES[name]


LOSSES = {
    "cross_entropy": cross_entropy
}

if __name__ == '__main__':
    tensor = torch.randn([4, 10])
    target = torch.randint(0, 10, [4])
    loss = cross_entropy(tensor, target)
    print(loss)
