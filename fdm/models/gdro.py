from __future__ import annotations
import logging

from ethicml import Dataset
import torch
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import trange

from fdm.models import Classifier

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class GDRO:
    def fit(
        self,
        classifier: Classifier,
        train_data: Dataset | DataLoader,
        epochs: int,
        device: torch.device,
        test_data: Dataset | DataLoader | None = None,
        pred_s: bool = False,
        batch_size: int = 256,
        test_batch_size: int = 1000,
        lr_milestones: dict | None = None,
        c_param: float = 1.0,
    ):
        if not isinstance(train_data, DataLoader):
            train_data = DataLoader(
                train_data, batch_size=batch_size, shuffle=True, pin_memory=True
            )
        if test_data is not None:
            if not isinstance(test_data, DataLoader):
                test_data = DataLoader(
                    test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True
                )

        scheduler = None
        if lr_milestones is not None:
            scheduler = MultiStepLR(optimizer=self.optimizer, **lr_milestones)

        LOGGER.info("Training classifier...")
        pbar = trange(epochs)
        for epoch in pbar:
            classifier.model.train()

            for x, s, y in train_data:

                if pred_s:
                    target = s
                    raise NotImplementedError("This wouldn't make sense.")
                else:
                    target = y

                x = x.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                classifier.optimizer.zero_grad()
                loss = []
                for _s in train_data.dataset.s.unique():
                    _loss, _acc = self.routine(
                        classifier, x[s == _s], target[s == _s], c_param=c_param
                    )
                    loss.append(_loss)

                # loss, acc = self.routine(classifier, x, target)
                max(loss).backward()
                classifier.step()

            if test_data is not None:

                classifier.eval()
                avg_test_acc = 0.0

                with torch.set_grad_enabled(False):
                    for x, s, y in test_data:

                        if pred_s:
                            target = s
                        else:
                            target = y

                        x = x.to(device)
                        target = target.to(device)

                        loss, acc = classifier.routine(x, target)
                        avg_test_acc += acc

                avg_test_acc /= len(test_data)

                pbar.set_postfix(epoch=epoch + 1, avg_test_acc=avg_test_acc)
            else:
                pbar.set_postfix(epoch=epoch + 1)

            if scheduler is not None:
                scheduler.step(epoch)
        pbar.close()

        return classifier

    def routine(
        self,
        classifier: Classifier,
        data: Tensor,
        targets: Tensor,
        instance_weights: Tensor | None = None,
        c_param: float = 1.0,
    ) -> tuple[Tensor, float]:
        """Classifier routine.

        Args:
            data: Tensor. Input data to the classifier.
            targets: Tensor. Prediction targets.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        outputs = classifier(data)
        loss = classifier.apply_criterion(outputs, targets)
        if instance_weights is not None:
            loss = loss.view(-1) * instance_weights.view(-1)
        loss = loss.mean()
        loss += c_param / torch.sqrt(torch.ones_like(loss) * data.shape[0])
        acc = classifier.compute_accuracy(outputs, targets)

        return loss, acc
