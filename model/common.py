import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, seq):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()


class BaseRGBModel(ABCModel):

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._get_params(), **opt_args), \
            torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)


def step(optimizer, scaler, loss, lr_scheduler=None, backward_only=False):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if not backward_only:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()


class SingleStageGRU(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=5):
        super(SingleStageGRU, self).__init__()
        self.backbone = nn.GRU(
            in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(2 * hidden_dim),
            nn.Dropout(),
            nn.Linear(2 * hidden_dim, out_dim))

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        x, _ = self.backbone(x)
        x = self.fc_out(x.reshape(-1, x.shape[-1]))
        return x.view(batch_size, clip_len, -1)


class SingleStageTCN(nn.Module):

    class DilatedResidualLayer(nn.Module):
        def __init__(self, dilation, in_channels, out_channels):
            super(SingleStageTCN.DilatedResidualLayer, self).__init__()
            self.conv_dilated = nn.Conv1d(
                in_channels, out_channels, 3, padding=dilation,
                dilation=dilation)
            self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
            self.dropout = nn.Dropout()

        def forward(self, x, mask):
            out = F.relu(self.conv_dilated(x))
            out = self.conv_1x1(out)
            out = self.dropout(out)
            return (x + out) * mask[:, 0:1, :]

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dilate):
        super(SingleStageTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([
            SingleStageTCN.DilatedResidualLayer(
                2 ** i if dilate else 1, hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x, m=None):
        batch_size, clip_len, _ = x.shape
        if m is None:
            m = torch.ones((batch_size, 1, clip_len), device=x.device)
        else:
            m = m.permute(0, 2, 1)
        x = self.conv_1x1(x.permute(0, 2, 1))
        for layer in self.layers:
            x = layer(x, m)
        x = self.conv_out(x) * m[:, 0:1, :]
        return x.permute(0, 2, 1)
