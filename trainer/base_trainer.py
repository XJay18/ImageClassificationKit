import math
import os
import random
import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import load_dataset
from loss import fetch_loss
from models import load_model
from models.common import freeze_weights
from optimizer import fetch_optimizer
from scheduler import fetch_scheduler
from trainer.utils import AccMeter, AverageMeter, Logger, Timer
from trainer.utils import MODELS_PATH, center_print


class BaseTrainer(object):
    def __init__(self, config, stage="Train"):
        feasible_stage = ["Train", "Test"]
        assert stage in feasible_stage, "The argument 'stage' is not in %s." % feasible_stage

        model_cfg = config.get("model", None)
        data_cfg = config.get("data", None)
        config_cfg = config.get("config", None)

        self.gpu = False
        self.device = config_cfg.get("device", None)
        if self.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)

        self.model_name = model_cfg["name"]

        if stage == "Train":
            # debug mode: no log dir, no train_val operation.
            self.debug = config_cfg["debug"]
            print("Using debug mode: %s." % self.debug)
            print("*" * 20)

            # load training dataset
            self.train_set = load_dataset(data_cfg["train"])(
                root=data_cfg["path"], train=True
            )
            # wrapped with data loader
            self.train_loader = DataLoader(self.train_set, batch_size=data_cfg["train_batch_size"])

            # load validation dataset
            self.val_set = load_dataset(data_cfg["val"])(
                root=data_cfg["path"], train=False
            )
            # wrapped with data loader
            self.val_loader = DataLoader(self.val_set, batch_size=data_cfg["val_batch_size"])

            self.resume = config_cfg.get("resume", False)

            if not self.debug:
                time_format = "%Y-%m-%d...%H.%M.%S"
                run_id = time.strftime(time_format, time.localtime(time.time()))
                self.run_id = config_cfg.get("id", run_id)
                self.dir = os.path.join("runs", self.model_name, self.run_id)

                if not self.resume:
                    if os.path.exists(self.dir):
                        raise ValueError("Error: given id '%s' already exists." % self.run_id)
                    os.makedirs(self.dir)
                    print("Writing config file to file directory: %s." % self.dir)
                    yaml.dump(config, open(os.path.join(self.dir, 'train_config.yml'), 'w'))
                else:
                    print("Resuming the history in file directory: %s." % self.dir)

                model_file = MODELS_PATH[self.model_name]
                target_model_file = os.path.join(self.dir, "model.py")
                os.system("cp " + model_file + " " + target_model_file)

                # redirect the std out stream
                sys.stdout = Logger(os.path.join(self.dir, 'records.txt'))
                print("Logging directory: %s." % self.dir)

                center_print('Train configurations begins.')
                pprint(config)
                center_print('Train configurations ends.')

            model_cfg.pop("name")
            self.model = load_model(self.model_name)(**model_cfg)
            optim_cfg = config_cfg["optimizer"]
            optim_name = optim_cfg["name"]
            optim_cfg.pop("name")
            self.optimizer = fetch_optimizer(optim_name)(self.model.parameters(), **optim_cfg)
            self.scheduler = fetch_scheduler(self.optimizer, config_cfg.get("scheduler", None))
            self.loss_func = fetch_loss()

            self.best_acc = 0.0
            self.best_step = 1
            self.start_step = 1

            # total number of steps (or epoch) to train
            self.num_steps = config_cfg["num_steps"]
            self.num_epoch = math.ceil(self.num_steps / len(self.train_loader))

            # the number of steps to write down a log
            self.log_steps = config_cfg["log_steps"]
            # the number of steps to validate on val dataset once
            self.val_steps = config_cfg["val_steps"]

            self.acc_meter = AccMeter()
            self.loss_meter = AverageMeter()

            if self.resume:
                self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

        if stage == "Test":
            # load test dataset
            self.test_set = load_dataset(data_cfg["test"])(
                root=data_cfg["path"], train=False
            )
            # wrapped with data loader
            self.test_loader = DataLoader(self.test_set, batch_size=data_cfg["batch_size"])
            self.run_id = config_cfg["id"]
            self.ckpt_fold = config_cfg.get("ckpt_fold", "runs")
            self.dir = os.path.join(self.ckpt_fold, self.model_name, self.run_id)

            model_cfg.pop("name")
            self.model = load_model(self.model_name)(**model_cfg)

            # redirect the std out stream
            sys.stdout = Logger(os.path.join(self.dir, "test_result.txt"))
            print('Run dir: {}'.format(self.dir))

            center_print('Test configurations begins')
            pprint(config)
            center_print('Test configurations ends')

            self._load_ckpt(best=True, train=False)

        if torch.cuda.is_available() and self.device is not None:
            print("Using cuda device: %d." % self.device)
            self.gpu = True
            self.model = self.model.cuda()
        else:
            print("Using cpu device.")
            self.device = torch.device("cpu")

    @staticmethod
    def fixed_randomness():
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _save_ckpt(self, step, best=False):
        save_dir = os.path.join(self.dir, "best_model.bin" if best else "latest_model.bin")
        torch.save({
            "step": step,
            "best_step": self.best_step,
            "best_acc": self.best_acc,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, save_dir)

    def _load_ckpt(self, best=False, train=False):
        load_dir = os.path.join(self.dir, "best_model.bin" if best else "latest_model.bin")
        load_dict = torch.load(load_dir, map_location=self.device)
        self.start_step = load_dict["step"]
        self.best_step = load_dict["best_step"]
        self.best_acc = load_dict["best_acc"]
        self.model.load_state_dict(load_dict["model"])
        if train:
            self.optimizer.load_state_dict(load_dict["optimizer"])
            self.scheduler.load_state_dict(load_dict["scheduler"])
        print("Loading checkpoint from %s, best step: %d, best acc: %.4f"
              % (load_dir, self.best_step, self.best_acc))

    def to_cuda(self, *args):
        return [obj.cuda() for obj in args]

    def train(self):
        timer = Timer()
        writer = None if self.debug else SummaryWriter(log_dir=self.dir)
        center_print("Training begins......")
        start_epoch = self.start_step // len(self.train_loader) + 1
        for epoch_idx in range(start_epoch, self.num_epoch + 1):
            self.acc_meter.reset()
            self.loss_meter.reset()
            self.optimizer.step()
            self.scheduler.step(epoch=epoch_idx)
            train_generator = tqdm(enumerate(self.train_loader, 1), position=0, leave=True)

            for batch_idx, data in train_generator:
                global_step = (epoch_idx - 1) * len(self.train_loader) + batch_idx
                self.model.train()
                I, Y = data
                if self.gpu:
                    (I, Y) = self.to_cuda(I, Y)
                Y_pre = self.model(I)
                loss = self.loss_func(Y_pre, Y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.acc_meter.update(Y_pre, Y)
                self.loss_meter.update(loss.item())

                if global_step % self.log_steps == 0 and writer is not None:
                    writer.add_scalar("train/acc", self.acc_meter.mean_acc(), global_step)
                    writer.add_scalar("train/loss", self.loss_meter.avg, global_step)
                    writer.add_scalar("train/lr", self.scheduler.get_lr()[0], global_step)

                train_generator.set_description(
                    "Train Epoch %d (%d/%d), Global Step %d, Loss %.4f, ACC %.4f, LR %.6f" % (
                        epoch_idx, batch_idx, len(self.train_loader), global_step,
                        self.loss_meter.avg, self.acc_meter.mean_acc(),
                        self.scheduler.get_lr()[0]
                    )
                )

                # validating process
                if global_step % self.val_steps == 0 and not self.debug:
                    print()
                    self.validate(epoch_idx, global_step, timer, writer)

                # when num_steps has been set and the training process will
                # be stopped earlier than the specified num_epochs, then stop.
                if self.num_steps is not None and global_step == self.num_steps:
                    if writer is not None:
                        writer.close()
                    print()
                    center_print("Training process ends.")
                    return

            train_generator.close()
            print()
        writer.close()
        center_print("Training process ends.")

    def validate(self, epoch, step, timer, writer):
        with torch.no_grad():
            acc = AccMeter()
            loss_meter = AverageMeter()
            val_generator = tqdm(enumerate(self.val_loader, 1), position=0, leave=True)
            for val_idx, data in val_generator:
                self.model.eval()
                I, Y = data
                if self.gpu:
                    (I, Y) = self.to_cuda(I, Y)
                Y_pre = self.model(I)
                loss = self.loss_func(Y_pre, Y)
                acc.update(Y_pre, Y)
                loss_meter.update(loss.item())

                val_generator.set_description(
                    "Eval Epoch %d (%d/%d), Global Step %d, Loss %.4f, ACC %.4f" % (
                        epoch, val_idx, len(self.val_loader), step,
                        loss_meter.avg, acc.mean_acc()
                    )
                )

            print("Eval Epoch %d, ACC %.4f" % (epoch, acc.mean_acc()))
            if writer is not None:
                writer.add_scalar("val/loss", loss_meter.avg, step)
                writer.add_scalar("val/acc", acc.mean_acc(), step)
            if acc.mean_acc() > self.best_acc:
                self.best_acc = acc.mean_acc()
                self.best_step = step
                self._save_ckpt(step, best=True)
            print("Best Step %d, Best ACC %.4f, Running Time: %s, Estimated Time: %s" % (
                self.best_step, self.best_acc, timer.measure(), timer.measure(step / self.num_steps)
            ))
            self._save_ckpt(step, best=False)

    def test(self):
        freeze_weights(self.model)
        t_idx = random.randint(1, len(self.test_loader) + 1)
        self.fixed_randomness()  # for reproduction

        acc = AccMeter()
        test_generator = tqdm(enumerate(self.test_loader, 1))
        categories = self.test_loader.dataset.categories
        for idx, data in test_generator:
            self.model.eval()
            I, Y = data
            if self.gpu:
                (I, Y) = self.to_cuda(I, Y)
            Y_pre = self.model(I)
            acc.update(Y_pre, Y)

            test_generator.set_description(
                "Test %d/%d, ACC %.4f" % (idx, len(self.test_loader), acc.mean_acc())
            )
            if idx == t_idx:
                images = I[:4]
                pred = Y_pre[:4]
                gt = Y[:4]
                self.plot_figure(images, pred, gt, 2, categories)

        print("Test, FINAL ACC %.4f" % acc.mean_acc())

    @staticmethod
    def plot_figure(images, pred, gt, nrow, categories=None):
        plot = make_grid(
            images, nrow, padding=4, normalize=True, scale_each=True, pad_value=1)
        pred = pred.argmax(1).cpu().numpy()
        gt = gt.cpu().numpy()
        if categories is not None:
            pred = [categories[i] for i in pred]
            gt = [categories[i] for i in gt]
        plot = plot.permute([1, 2, 0])
        plot = plot.cpu().numpy()
        plt.figure()
        plt.imshow(plot)
        plt.title("pred: %s\ngt: %s" % (pred, gt))
        plt.axis("off")
        plt.show()
