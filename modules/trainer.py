DEBUG = False

import os
from abc import abstractmethod
import time
import torch
import pandas as pd
from numpy import inf


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler=None):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)  # 如果想禁用 early_stop，可在命令行设大值

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"\n=== Epoch {epoch} / {self.epochs} ===")
            result = self._train_epoch(epoch)

            # 每个epoch结束后更新学习率
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                print(f"Learning rate after epoch {epoch}: {self.optimizer.param_groups[0]['lr']}")

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print(f"Warning: Metric '{self.mnt_metric}' not found. Monitoring disabled.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                # 打印 not improved count，用于调试早停
                print(f"Validation metric: {log.get(self.mnt_metric, None)}, Best: {self.mnt_best}, Not improved count: {not_improved_count}")

                if not_improved_count > self.early_stop:
                    print(f"Early stop triggered after {self.early_stop} epochs without improvement.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir,
                                   self.args.dataset_name + '.csv')

        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)

        new_rows = pd.DataFrame([self.best_recorder['val'], self.best_recorder['test']])
        record_table = pd.concat([record_table, new_rows], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: No GPU available, training on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(f"Warning: configured {n_gpu_use} GPUs, but only {n_gpu} available.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print(f"Saved checkpoint: {filename}")
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saved current best model: model_best.pth")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.lr_scheduler and 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print(f"Checkpoint loaded. Resume from epoch {self.start_epoch}")

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print(f"Best results (w.r.t {self.args.monitor_metric}) in validation set:")
        for key, value in self.best_recorder['val'].items():
            print(f"\t{key:15s}: {value}")
        print(f"Best results (w.r.t {self.args.monitor_metric}) in test set:")
        for key, value in self.best_recorder['test'].items():
            print(f"\t{key:15s}: {value}")


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler,
                 train_dataloader, val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        train_loss = 0.0
        n_iter = 0

        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)

            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)

            train_loss += loss.item()
            n_iter += 1

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        avg_train_loss = train_loss / n_iter if n_iter > 0 else 0.0
        log = {'train_loss': avg_train_loss}

        # 每 epoch 验证集+测试集评估
        self.model.eval()
        with torch.no_grad():
            # 验证集
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)

                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update({f'val_{k}': v for k, v in val_met.items()})

            # 测试集
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)

                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update({f'test_{k}': v for k, v in test_met.items()})

        return log
