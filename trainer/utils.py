import time
import sys
import numpy as np

# Tracking the path to the definition of the model.
MODELS_PATH = {
    "ResNetMini": "models/resnet.py",
    "ResNextMini": "models/resnext.py",
    "DenseNetMini": "models/densenet.py",
    "DenseNet121": "models/densenet.py",
}


def center_print(content, around='*', repeat_around=10):
    num = repeat_around
    s = around
    print(num * s + ' %s ' % content + num * s)


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class Timer(object):
    """The class for timer."""

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


class AccMeter(object):
    def __init__(self):
        self.nums = 0
        self.acc = 0

    def reset(self):
        self.nums = 0
        self.acc = 0

    def update(self, pred, target):
        pred = pred.argmax(1).cpu().numpy()
        target = target.cpu().numpy()
        self.nums += target.shape[0]
        self.acc += np.sum(pred == target)

    def mean_acc(self):
        return self.acc / self.nums


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
