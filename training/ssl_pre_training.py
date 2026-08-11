import lightning.pytorch as pl
import torch
import torch.optim as optim
from torch.nn import functional as F
import logging
import gc

from training.models.projection_heads import ProjectionHeadSimCLR

"""Model setup for self-supervised SimCLR pre-training"""

class PatchMatchSSL(pl.LightningModule):
    def __init__(self, left_encoder, right_encoder, shared_backbone=None, ssl_mode: str = 'combined', **kwargs):
        super().__init__()
        self.loss_mode = ssl_mode #combined/sequential or individual per branch
            # Code was prepared to pre-train models that share the backbone.
            #  In that case, loss can be backpropagation sequentially or combined (averaged).
        self.save_hyperparameters(ignore=['left_encoder', 'right_encoder', 'shared_backbone'])

        # f(.)
        self.left_encoder = left_encoder if ssl_mode != "S2" else None
        self.right_encoder = right_encoder if ssl_mode != "PS" else None
        self.shared_backbone = shared_backbone

        # g(.)
        if self.left_encoder and self.right_encoder:
            out_dim = getattr(shared_backbone, "out_dim", None) or right_encoder.out_dim if right_encoder.out_dim == left_encoder.out_dim else None
        else:
            out_dim = getattr(shared_backbone, "out_dim", None) or getattr(right_encoder, "out_dim", None) or getattr(left_encoder, "out_dim", None)
        self.projection_head = ProjectionHeadSimCLR(input_dim=out_dim, output_dim=self.hparams.hidden_dim)

        self.current_side = 'left' if ssl_mode in ['PS', 'sequential'] else 'right' if ssl_mode == 'S2' else None
        self.hparams.max_epochs = self.hparams.max_epochs * 2 if ssl_mode == "sequential" else self.hparams.max_epochs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        if self.current_side == "left":
            z = self.left_encoder(x)
        elif self.current_side == "right":
            z = self.right_encoder(x)
        else:
            raise NotImplementedError

        if self.shared_backbone:
            z = self.shared_backbone.forward(z)

        z = self.projection_head(z)
        return z

    def loss(self, batch, mode):
        imgs = torch.cat(batch, dim=0)
        imgs = imgs.to(self.device)

        with torch.set_grad_enabled(self.training):
            feats = self(imgs)
            del imgs

            cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)

            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

            cos_sim = cos_sim / self.hparams.temperature
            loss = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            loss = loss.mean()

            self.log(f'{mode}_{self.current_side}_loss', loss.detach(), batch_size=batch[0].shape[0])
            self.log(f'{mode}_loss', loss.detach(), batch_size=batch[0].shape[0])

        with torch.no_grad():
            comb_sim = torch.cat([cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

            self.log(f'{mode}_{self.current_side}_acc_top1', (sim_argsort == 0).float().mean(), batch_size=batch[0].shape[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{mode}_{self.current_side}_acc_top5', (sim_argsort < 5).float().mean(), batch_size=batch[0].shape[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{mode}_{self.current_side}_acc_mean_pos', 1 + sim_argsort.float().mean(), batch_size=batch[0].shape[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{mode}_acc_top3', (sim_argsort < 3).float().mean(), batch_size=batch[0].shape[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{mode}_acc_top5', (sim_argsort < 5).float().mean(), batch_size=batch[0].shape[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clean up intermediate tensors
        del feats, comb_sim, sim_argsort
        torch.cuda.empty_cache()

        return loss

    def _step(self, batch, batch_idx, mode):
        if self.loss_mode == 'combined':
            loss_val = []
            if batch['left'] != None:
                left_imgs = batch['left']
                self.current_side = 'left'
                loss_val.append(self.loss(left_imgs, mode=mode))
                del left_imgs
                torch.cuda.empty_cache()

            if batch['right'] != None:
                right_imgs = batch['right']
                self.current_side = 'right'
                loss_val.append(self.loss(right_imgs, mode=mode))
                del right_imgs
                torch.cuda.empty_cache()

            return sum(loss_val) / len(loss_val)

        elif self.loss_mode == "sequential":
            side, batch = batch
            self.current_side = side
            loss = self.loss(batch, mode=mode)
            del batch
            torch.cuda.empty_cache()
            return loss
        else:
            loss = self.loss(batch, mode=mode)
            torch.cuda.empty_cache()
            return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="validate")

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"{name} has no gradients! While on {self.current_side}")

        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        """Save model weights after certain amounts of epochs"""
        if self.current_epoch in [0,1,2,3,4,9,14,19,24]:
            platform = "PS" if self.current_side == "left" else "S2"
            path = f'output/weights/SSL_{platform}_harsh_ep={self.current_epoch + 1}.ckpt'
            torch.save(self.state_dict(), path)
            logging.info(f"Model saved at {path}")