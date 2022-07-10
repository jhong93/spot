""" Various wrapped models """

from contextlib import nullcontext
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import step, ABCModel, SingleStageGRU, SingleStageTCN
from .impl.gtad import GCNeXt
from .impl.calf import get_calf


class FeatureModel(ABCModel):

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._get_params(), **opt_args), \
            torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        return self._model.state_dict()

    def load(self, state_dict):
        self._model.load_state_dict(state_dict)


def _epoch_helper(
    loader,
    optimizer,
    scaler,
    lr_scheduler,
    model,
    num_classes,
    device,
    fg_weight=5,
    calf_weight=0.001
):
    if optimizer is None:
        model.eval()
    else:
        optimizer.zero_grad()
        model.train()

    ce_kwargs = {}
    ce_kwargs['weight'] = torch.FloatTensor(
        [1] + [fg_weight] * (num_classes - 1)).to(device)

    epoch_loss = 0.
    with torch.no_grad() if optimizer is None else nullcontext():
        for batch in tqdm(loader):
            x = batch['feature'].float().to(device)
            y = batch['label'].to(device)
            m = batch['mask'].to(device)

            if 'calf' in batch:
                c = batch['calf'].to(device)

            batch_size = x.shape[0]
            with (
                    nullcontext() if scaler is None
                    else torch.cuda.amp.autocast()
            ):
                pred = model(x, m)
                if len(pred.shape) == 3:    # Single stage only
                    pred = pred.unsqueeze(0)
                loss = 0.
                for i in range(pred.shape[0]):
                    pred_i = pred[i] * m.unsqueeze(2)
                    loss += F.cross_entropy(
                        pred_i.reshape(
                            batch_size * y.shape[1], num_classes),
                        y.flatten(), **ce_kwargs)

                    if 'calf' in batch:
                        loss += calf_weight * get_calf(pred_i, c)

            if optimizer is not None:
                step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)
            epoch_loss += loss.detach().cpu().item() / len(loader)
    return epoch_loss


class GRU(FeatureModel):

    class Impl(nn.Module):

        def __init__(self, feat_dim, num_classes, num_stages, hidden_dim=128):
            super(GRU.Impl, self).__init__()

            self.stage1 = SingleStageGRU(feat_dim, hidden_dim, num_classes)
            self.stages = None
            if num_stages > 1:
                self.stages = nn.ModuleList([
                    SingleStageGRU(num_classes, hidden_dim, num_classes)
                    for _ in range(num_stages - 1)])

        def forward(self, x, m):
            x = self.stage1(x)
            if self.stages is None:
                return x
            else:
                outputs = [x]
                for stage in self.stages:
                    x = stage(F.softmax(x, dim=2))
                    outputs.append(x)
                return torch.stack(outputs, dim=0)

    def __init__(self, feat_dim, num_classes, num_stages=1, device='cuda'):
        self.device = device
        self._model = GRU.Impl(feat_dim, num_classes, num_stages)
        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        return _epoch_helper(
            loader, optimizer, scaler, lr_scheduler, self._model,
            self._num_classes, self.device)

    def predict(self, seq):
        seq_len = seq.shape[0]
        if not isinstance(seq, torch.FloatTensor):
            seq = torch.FloatTensor(seq)
        seq = seq.unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(seq, None)
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = F.softmax(pred.view(seq_len, -1), dim=1)
            pred_cls = torch.argmax(pred, axis=1)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


class TCN(FeatureModel):

    class Impl(nn.Module):

        def __init__(self, feat_dim, num_classes, num_stages,
                     hidden_dim=256, num_layers=5, dilate=True):
            super(TCN.Impl, self).__init__()
            self.stage1 = SingleStageTCN(
                feat_dim, hidden_dim, num_classes, num_layers, dilate)
            self.stages = None
            if num_stages > 1:
                self.stages = nn.ModuleList([SingleStageTCN(
                    num_classes, hidden_dim, num_classes, num_layers, dilate)
                    for _ in range(num_stages - 1)])

        def forward(self, x, m):
            m = m.unsqueeze(2)
            x = self.stage1(x, m)
            if self.stages is None:
                return x
            else:
                outputs = [x]
                for stage in self.stages:
                    x = stage(F.softmax(x, dim=2) * m[:, 0:1, :], m)
                    outputs.append(x)
                return torch.stack(outputs, dim=0)

    def __init__(self, feat_dim, num_classes, num_stages=1, device='cuda'):
        self.device = device
        self._feat_dim = feat_dim
        self._model = TCN.Impl(feat_dim, num_classes, num_stages)
        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        return _epoch_helper(
            loader, optimizer, scaler, lr_scheduler, self._model,
            self._num_classes, self.device)

    def predict(self, seq):
        seq_len = seq.shape[0]
        if not isinstance(seq, torch.FloatTensor):
            seq = torch.FloatTensor(seq)
        seq = seq.unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(seq, torch.ones(1, seq_len, device=self.device))
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = F.softmax(pred.reshape(seq_len, -1), dim=1)
            pred_cls = torch.argmax(pred, axis=1)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


class GCN(FeatureModel):

    class Impl(nn.Module):

        def __init__(self, feat_dim, num_classes, hidden_dim=256, num_layers=2):
            super(GCN.Impl, self).__init__()

            self.idx_list = []
            self.fc_in = nn.Linear(feat_dim, hidden_dim)
            gcn_layers = [GCNeXt(hidden_dim, hidden_dim, k=3, groups=32,
                          idx=self.idx_list) for _ in range(num_layers)]
            self.backbone = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                          groups=4),
                nn.ReLU(inplace=True),
                *gcn_layers
            )
            self.dropout = nn.Dropout()
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x, m):
            del self.idx_list[:]
            batch_size, clip_len, _ = x.shape
            x = self.fc_in(x.view(batch_size * clip_len, -1))
            x = F.relu(x).view(batch_size, clip_len, -1)
            x = self.backbone(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)
            return self.fc(x.reshape(batch_size * clip_len, -1)).view(
                batch_size, clip_len, -1)

    def __init__(self, feat_dim, num_classes, device='cuda'):
        self.device = device
        self._model = GCN.Impl(feat_dim, num_classes)
        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        return _epoch_helper(
            loader, optimizer, scaler, lr_scheduler, self._model,
            self._num_classes, self.device)

    def predict(self, seq):
        seq_len = seq.shape[0]
        if not isinstance(seq, torch.FloatTensor):
            seq = torch.FloatTensor(seq)
        seq = seq.unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(seq, None)
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = F.softmax(pred.view(seq_len, -1), dim=1)
            pred_cls = torch.argmax(pred, axis=1)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


class ASFormer(FeatureModel):

    class Impl(nn.Module):

        def __init__(self, feat_dim, num_classes, num_f_maps=64, num_layers=10,
                     num_decoders=3):
            super(ASFormer.Impl, self).__init__()

            from .impl.asformer import MyTransformer

            r1, r2 = 2, 2
            self._net = MyTransformer(
                num_decoders, num_layers, r1, r2, num_f_maps, feat_dim,
                num_classes, channel_masking_rate=0.3)

        def forward(self, x, m):
            return self._net(x.permute(0, 2, 1), m.unsqueeze(1)).permute(
                0, 1, 3, 2)

    def __init__(self, feat_dim, num_classes, device='cuda'):
        super(ASFormer, self).__init__()
        self.device = device
        self._model = ASFormer.Impl(feat_dim, num_classes)
        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        return _epoch_helper(
            loader, optimizer, scaler, lr_scheduler, self._model,
            self._num_classes, self.device)

    def predict(self, seq):
        seq_len = seq.shape[0]
        if not isinstance(seq, torch.FloatTensor):
            seq = torch.FloatTensor(seq)
        seq = seq.unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(seq, torch.ones(1, seq_len, device=self.device))
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = F.softmax(pred.reshape(seq_len, -1), dim=1)
            pred_cls = torch.argmax(pred, axis=1)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()