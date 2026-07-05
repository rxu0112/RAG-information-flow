import torch
from torch import nn
from Other_Baselines.ragu.passage_utility.swag import utils
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
# from transformers import get_linear_schedule_with_warmup
import numpy as np
from Other_Baselines.ragu.passage_utility.dataset_collection import SESingleDataset, PosNegSingleDataset, SEPairwiseDataset
from torch.utils.data import DataLoader
from tqdm import trange
from torch import optim
import os

# def collate_fn(batch):
#     input_ids1 = [item['input_ids1'] for item in batch]
#     input_ids2 = [item['input_ids2'] for item in batch]
#     attention_mask1 = [item['attention_mask1'] for item in batch]
#     attention_mask2 = [item['attention_mask2'] for item in batch]
#     token_type_ids1 = [item['token_type_ids1'] for item in batch]
#     token_type_ids2 = [item['token_type_ids2'] for item in batch]
#     score1 = [item['score1'] for item in batch]
#     score2 = [item['score2'] for item in batch]
#     acc1 = [item['acc1'] for item in batch]
#     acc2 = [item['acc2'] for item in batch]
#     targets = [item['targets'] for item in batch]

#     # 使用 pad_sequence 来确保 input_ids 和 attention_mask 都填充到相同长度
#     input_ids1 = pad_sequence(input_ids1, batch_first=True, padding_value=0)
#     input_ids2 = pad_sequence(input_ids2, batch_first=True, padding_value=0)
#     attention_mask1 = pad_sequence(attention_mask1, batch_first=True, padding_value=0)
#     attention_mask2 = pad_sequence(attention_mask2, batch_first=True, padding_value=0)
#     token_type_ids1 = pad_sequence(token_type_ids1, batch_first=True, padding_value=0)
#     token_type_ids2 = pad_sequence(token_type_ids2, batch_first=True, padding_value=0)

#     # 将 labels 合并成一个 tensor
#     return {
#         'input_ids1': input_ids1,
#         'input_ids2': input_ids2,
#         'attention_mask1': attention_mask1,
#         'attention_mask2': attention_mask2,
#         'token_type_ids1': token_type_ids1,
#         'token_type_ids2': token_type_ids2,
#         'targets': torch.stack(targets),
#         'score1': torch.stack(score1),
#         'score2': torch.stack(score2),
#         'acc1': torch.stack(acc1),
#         'acc2': torch.stack(acc2),
#     }

class VanillaBert(nn.Module):
    def __init__(self, base_model, lr, epochs, pretrained_model, ilr, device, 
        weight_decay, prior_var=1, margin=0.1, combine_loss=None, weight_rank=1, 
        weight_aux=0, model_select='combined') -> None:
        super().__init__()
        self.base_model = base_model
        self.base_model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.pretrained_model = pretrained_model
        self.prior_var = prior_var
        self.optimizer = AdamW(self.base_model.parameters(),
                               lr=self.lr, weight_decay=weight_decay)
        self.update_opt = optim.SGD(
            self.base_model.parameters(), lr=ilr, momentum=0.9
            , weight_decay=weight_decay)
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        self.mse_loss = nn.MSELoss()
        self.be_loss = nn.BCEWithLogitsLoss()
        self.combined_mse = (combine_loss == 'mse')
        self.combined_be = (combine_loss == 'be')
        self.a_w = weight_aux
        self.r_w = weight_rank
        self.device = device
        self.model_select = model_select


    def train(self, data_loader, valid_loader, save_dir, stop_epochs, save_name, checkpoint, log_file):
        val_acc, save_epoch, start_epoch = 0, 0, 0
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=len(
                data_loader) * self.epochs * 0.1,
            num_training_steps=len(data_loader) * self.epochs)
        if checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, self.epochs):
            print(f'---- epoch: {epoch}/{self.epochs} ----')
            log_file.write(f'---- epoch: {epoch}/{self.epochs} ----')
            utils.train_epoch(data_loader,
                              self.base_model,
                              self._loss,
                              self.optimizer,
                              verbose=True,
                              regularizer=None,
                              scheduler=self.scheduler,
                              criteria=self.model_select,
                              single_net=(self.r_w==0))
            if valid_loader:
                metrics = utils.eval(valid_loader, self.base_model, self._loss, criteria=self.model_select, single_net=(self.r_w==0))
                print('validation', metrics)
                log_file.write(f'validation {metrics}')
                if metrics['accuracy'] > val_acc:
                # if metrics['loss'] > val_acc:
                    print('finding better model')
                    save_epoch = epoch
                    val_acc = metrics['accuracy']
                    utils.save_checkpoint(save_dir, epoch=epoch, name=save_name+'best-val-acc-model', state_dict=self.state_dict(
                    ), optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict())
                    print(
                        f'save best model to {os.path.join(save_dir, save_name+"best-val-acc-model")}')
                    log_file.write(f'save best model to {os.path.join(save_dir, save_name+"best-val-acc-model")}')
                if epoch - save_epoch > stop_epochs:
                    print(
                        f'stopping without any improvement after {stop_epochs}')
                    log_file.write(f'stopping without any improvement after {stop_epochs}')
                    break

        utils.save_checkpoint(save_dir, epoch=epoch, name=save_name, state_dict=self.state_dict(
                ), optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict())
        print(f'save {epoch} model to {os.path.join(save_dir, save_name)}')
        log_file.write(f'save {epoch} model to {os.path.join(save_dir, save_name)}')

    def _loss(self, model, target, **kwargs):
        """target is (ranking, mse, mse, bce, bce)"""
        # standard cross-entropy loss function
        output = model(**kwargs)
        output = torch.cat(output, axis=0)
        # loss = F.cross_entropy(output, target)
        if self.r_w > 0:
            loss = self.r_w * self.loss_fn(output[0, :], output[1, :], target[0].to(self.device))
        if self.combined_be:
            if self.r_w > 0:
                l0 = self.be_loss(output[0, :], target[3].to(self.device))
                l1 = self.be_loss(output[1, :], target[4].to(self.device))
                loss += (self.a_w * (l0 + l1))
            else:
                l0 = self.be_loss(output[0, :], target[3].to(self.device))
                loss = self.a_w * l0 
        elif self.combined_mse:
            if self.r_w > 0:
                l0 = self.mse_loss(output[0, :], target[1].to(self.device))
                l1 = self.mse_loss(output[1, :], target[2].to(self.device))
                loss += (self.a_w * (l0 + l1))
            else:
                l0 = self.mse_loss(output[0, :], target[1].to(self.device))
                loss = self.a_w * l0
        return loss, output, {}

    def _regularizer(self, data_loader, precision=None):
        if precision is not None:
            def regularizer(model): return precision/2*sum([
                p.norm()**2 for p in model.parameters()
            ])
        else:
            regularizer = None
        return regularizer

    def predict(self, test_data, question, samples_nums, eval):
        # dataset = SESingleDataset(log_data=test_data, question=question, pretrained_model=self.pretrained_model)
        dataset = PosNegSingleDataset(
            log_data=test_data, question=question, pretrained_model=self.pretrained_model)
        data_loader = DataLoader(dataset, batch_size=512, shuffle=False)

        self.base_model.train()
        if eval:
            samples_nums = 1

        all_scores = np.zeros((len(dataset), samples_nums))

        for i in trange(samples_nums):
            scores = utils.predict(data_loader, self.base_model, eval)
            all_scores[:, i] = scores
            
        self.mean, self.var = np.mean(
            all_scores, axis=1), np.var(all_scores, axis=1)

        return self.mean, self.var, all_scores

    def get_utilities(self, test_data, question, sample_nums=1):
        mean, _ , alls_score= self.predict(test_data, question, sample_nums, eval=True)
        return mean, alls_score

    def update(self, data, stop_epochs):
        dataset = SEPairwiseDataset(
            log_data=data, pretrained_model=self.pretrained_model)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        min_loss = float('inf')

        for epoch in range(self.epochs):
            metric = utils.train_epoch(data_loader,
                                       self.base_model,
                                       self._loss,
                                       self.update_opt,
                                       verbose=False,
                                       regularizer=None,
                                       scheduler=None,
                                       criteria=self.model_select,
                                       single_net=(self.r_w==0))
            if metric['loss'] < min_loss:
                end_epoch = epoch
                min_loss = metric['loss']
            if (metric['accuracy'] == 100.0) or (epoch - end_epoch > stop_epochs):
                print('no improvement stopping...')
                print(f'----update epoch: {epoch+1}/{self.epochs}, {metric} ----')
                break
            print(f'----update epoch: {epoch+1}/{self.epochs}, {metric} ----')
