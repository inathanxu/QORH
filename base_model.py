import torch
import numpy as np
import time
import logging

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import RED_GNN_induc
from utils import cal_ranks, cal_performance
from tqdm import tqdm


class BaseModel(object):
    def __init__(self, args, loader):
        self.model = RED_GNN_induc(args, loader)

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = getattr(loader, 'n_ent_ind', loader.n_ent)
        self.n_batch = args.n_batch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.n_layer = args.n_layer

        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.lamb
        )
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0

    def train_batch(self, ddp_model=None):
        epoch_loss = 0
        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)

        t_time = time.time()

        model_to_train = ddp_model if ddp_model is not None else self.model
        model_to_train.train()

        iterator = tqdm(range(n_batch), desc="Training Batches",
                        position=1, leave=False, dynamic_ncols=True)

        for i in iterator:
            start = i * batch_size
            end = min(self.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.optimizer.zero_grad()

            scores = model_to_train(triple[:, 0], triple[:, 1], mode='train')

            device = scores.device

            pos_scores = scores[
                torch.arange(len(scores), device=device),
                torch.tensor(triple[:, 2], dtype=torch.long, device=device)
            ]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(
                -pos_scores +
                max_n +
                torch.log(torch.sum(torch.exp(scores - max_n), 1))
            )

            loss.backward()
            self.optimizer.step()

            for p in model_to_train.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)

            epoch_loss += loss.item()

            iterator.set_postfix({'loss': f"{loss.item():.4f}"})

        self.scheduler.step()
        self.t_time += time.time() - t_time

        v_mrr, t_mrr, out_str = self.evaluate(
            model_to_eval=model_to_train, split='valid'
        )
        return v_mrr, t_mrr, out_str

    def _evaluate_split(self, model, split):
        if split == 'valid':
            n_data = self.n_valid
            filters_by_query = self.loader.val_filters
            progress_desc = 'Validating'
        else:
            n_data = self.n_test
            filters_by_query = self.loader.tst_filters
            progress_desc = 'Testing'

        if n_data <= 0:
            raise ValueError(f"{split} set is empty, cannot compute ranking metrics")

        batch_size = self.n_batch
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        iterator = tqdm(
            range(n_batch), desc=progress_desc,
            position=1, leave=False, dynamic_ncols=True
        )

        for i in iterator:
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data=split)

            with torch.no_grad():
                scores = model(subs, rels, mode=split).detach().cpu().numpy()

            candidate_count = scores.shape[1]
            if objs.shape != scores.shape:
                raise ValueError(
                    f"{split} answer matrix shape {objs.shape} does not match "
                    f"model score shape {scores.shape}; please check the "
                    f"train/inference entity mapping"
                )

            filters = np.zeros(
                (len(subs), candidate_count), dtype=np.float32
            )
            for row, (sub, rel) in enumerate(zip(subs, rels)):
                filtered_entities = filters_by_query.get(
                    (int(sub), int(rel)), []
                )
                if len(filtered_entities) > 0:
                    filtered_entities = np.asarray(
                        filtered_entities, dtype=np.int64
                    )
                    if (filtered_entities.min() < 0 or
                            filtered_entities.max() >= candidate_count):
                        raise ValueError(
                            f"{split} filter contains entity IDs outside "
                            f"the candidate space [0, {candidate_count})"
                        )
                    filters[row, filtered_entities] = 1.0

            ranking += cal_ranks(scores, objs, filters)

        ranking = np.asarray(ranking)
        if ranking.size == 0:
            raise ValueError(
                f"{split} set has no computable correct answers, "
                f"cannot produce MRR"
            )
        return cal_performance(ranking)

    def evaluate(self, model_to_eval=None, split='both'):
        if split not in {'valid', 'test', 'both'}:
            raise ValueError(
                f"split must be 'valid', 'test', or 'both', got: {split!r}"
            )

        model = model_to_eval if model_to_eval is not None else self.model
        model.eval()
        i_time = time.time()
        v_mrr = t_mrr = None
        result_parts = []

        if split in {'valid', 'both'}:
            v_mrr, v_h1, v_h10 = self._evaluate_split(model, 'valid')
            result_parts.append(
                '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f' %
                (v_mrr, v_h1, v_h10)
            )

        if split in {'test', 'both'}:
            t_mrr, t_h1, t_h10 = self._evaluate_split(model, 'test')
            result_parts.append(
                '[TEST] MRR:%.4f H@1:%.4f H@10:%.4f' %
                (t_mrr, t_h1, t_h10)
            )

        i_time = time.time() - i_time

        result_parts.append(
            '[TIME] train:%.4f inference:%.4f' % (self.t_time, i_time)
        )
        out_str = '\t '.join(result_parts) + '\n'

        return v_mrr, t_mrr, out_str
