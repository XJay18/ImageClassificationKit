from thop import profile
from thop import clever_format
import torch
import argparse

from models import load_model
from trainer.utils import center_print


def arg_parser():
    parser = argparse.ArgumentParser(
        description="This is a script to help you get "
                    "the statistics information of "
                    "a specific network."
    )
    parser.add_argument("-d", "--device",
                        type=int,
                        default=0,
                        help="Specify the device to run the model.")
    parser.add_argument("-m", "--model",
                        type=str,
                        default="ResNetMini",
                        help="Specify the model for statistics.\n")

    return parser.parse_args()


if __name__ == '__main__':
    """  This file is for obtaining the statistics of the networks.  """

    def statistics():
        arg = arg_parser()
        name = arg.model
        d = torch.device(arg.device)
        print("Using device: %s." % d)

        model = load_model(name)
        model = model().cuda(d)

        print("\n", model, "\n")

        I = torch.randn([4, 3, 32, 32]).cuda(d)
        print("Testing tensor shape: {}.".format(I.shape))
        macs, params = profile(model, inputs=[I])
        macs, params = clever_format([macs, params], "%.2f")
        center_print("Statistics Information")
        print("Model: {}".format(name))
        print("MACs(G): {}".format(macs))
        print("Params(M): {}".format(params))
        center_print("Ends")


    statistics()
