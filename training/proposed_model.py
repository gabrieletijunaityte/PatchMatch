import logging
import random
from math import ceil, floor

import lightning.pytorch as pl
import pandas as pd
from torch import optim

from training.models.losses import *
from training.models.projection_heads import ProjectionHeadSimCLR, ProjectionHeadLinear
from training.models.similarities import CosSimilarity, DeepRelationalSimilarity

"""Proposed model implementation (capable to run all experimental setups)"""


class ProposedModel(pl.LightningModule):
    def __init__(self, left_encoder, right_encoder, shared_backbone=None, ssl_path=None, weight_decay = 1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['left_encoder', 'right_encoder', 'shared_backbone'])
        self.weight_decay = weight_decay

        # f(.)
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.shared_backbone = shared_backbone

        # g(.)
        if self.hparams.projection_head == "simclr":
            input_dim = shared_backbone.out_dim if shared_backbone else right_encoder.out_dim if right_encoder.out_dim == left_encoder.out_dim else None
            self.projection_head = ProjectionHeadSimCLR(input_dim=input_dim, output_dim=self.hparams.hidden_dim)
        elif self.shared_backbone and self.hparams.projection_head == "linear":
            self.projection_head = ProjectionHeadLinear(input_dim=self.shared_backbone.out_dim, output_dim=self.hparams.hidden_dim)
        else:
            self.projection_head = None

        # Similarity
        self.similarity = (
            CosSimilarity() if self.hparams.similarity_mode in ["Cosine_similarity", "NT-Xent", "NT-Xent_v2"] else
            DeepRelationalSimilarity() if self.hparams.similarity_mode == "Deep_Relational_Similarity" else
            None)
        if self.similarity is None:
            raise ValueError('Invalid similarity mode')

        # Loss
        self.loss = (
            CosineEmbeddingLoss(self.hparams.loss_margin) if self.hparams.loss_mode == "Cosine_Embedding_loss" else
            NTXentLossIn(temperature=self.hparams.loss_temp) if self.hparams.loss_mode == "NT-Xent_In" else
            NTXentLossOut(temperature=self.hparams.loss_temp) if self.hparams.loss_mode == "NT-Xent_Out" else
            NTXentLossFull(temperature=self.hparams.loss_temp) if self.hparams.loss_mode == "NT-Xent_full" else
            ClipLoss(temperature=self.hparams.loss_temp) if self.hparams.loss_mode == "Clip_loss" else
            DeepRelationalSimilarityLoss() if self.hparams.loss_mode == "Relational_NN_loss" else
            None)

        if self.loss is None:
            raise ValueError('Invalid loss mode')

        # If model is pre-trained through SSL, load it in
        if ssl_path:
            self.load_ssl_weights(ssl_path)

        if self.hparams.freeze:
            self.freezer()


    def get_ssl_weights(self, ssl_path):
        """Gets pre-trained SSL weights"""
        # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#nn-module-from-checkpoint
        if len(ssl_path) == 2:
            ssl_path_PS = [i for i in ssl_path if 'PS' in i][0]
            ssl_path_S2 = [i for i in ssl_path if 'S2' in i][0]

            checkpoint = torch.load(ssl_path_PS, map_location=self.device)
            left_current_params = [name for name, params in self.left_encoder.named_parameters()]

            if left_current_params[0].startswith('0.net'):
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                left_encoder_weights = {k.replace("left_encoder.", "0."): v for k, v in checkpoint.items()}

            checkpoint = torch.load(ssl_path_S2, map_location=self.device)
            right_current_params = [name for name, params in self.right_encoder.named_parameters()]

            if right_current_params[0].startswith('0.net'):
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                right_encoder_weights = {k.replace("right_encoder.", "0."): v for k, v in checkpoint.items()}

            backbone_weights = None
        else:
            checkpoint = torch.load(ssl_path[0], map_location=self.device)
            left_encoder_weights = {k.replace("left_encoder.", "0."): v for k, v in checkpoint["state_dict"].items() if
                                    k.startswith("left_encoder.")}
            right_encoder_weights = {k.replace("right_encoder.", ""): v for k, v in checkpoint["state_dict"].items() if
                                     k.startswith("right_encoder.")}
            backbone_weights = {k.replace("shared_backbone.", ""): v for k, v in checkpoint['state_dict'].items() if
                                k.startswith("shared_backbone.")}

        return left_encoder_weights, right_encoder_weights, backbone_weights

    def freeze_part(self, part):
        """Freeze model part"""
        for param in part.parameters():
            param.requires_grad = False

    def freezer(self):
        """Freeze based on configuration"""
        if self.hparams.freeze == 'encoders' or self.hparams.freeze == 'all':
            self.freeze_part(self.left_encoder)
            self.freeze_part(self.right_encoder)
            logging.info(f'Encoders frozen.')
        if (self.hparams.freeze == 'backbone' or self.hparams.freeze == 'all') and self.shared_backbone:
            self.freeze_part(self.shared_backbone)
            logging.info(f'Backbone frozen.')
        if self.hparams.freeze == "temp" and self.hparams.loss_mode == "Clip_loss":
            self.freeze_part(self.loss)

    def load_ssl_weights(self, ssl_path):
        """Loads pre-trained SSL weights"""
        left_encoder_weights, right_encoder_weights, backbone_weights = self.get_ssl_weights(ssl_path)

        missing_keys, unexpected_keys = self.left_encoder.load_state_dict(left_encoder_weights, strict=False)
        if missing_keys:
            logging.info(f"The following keys are missing from the pretrained model for left encoder: {missing_keys}")
        if unexpected_keys:
            logging.info(f"The following keys are unexpected from the pretrained model for left encoder:{unexpected_keys}")


        missing_keys, unexpected_keys = self.right_encoder.load_state_dict(right_encoder_weights, strict=False)
        if missing_keys:
            logging.info(f"The following keys are missing from the pretrained model for right encoder: {missing_keys}")
        if unexpected_keys:
            logging.info(f"The following keys are unexpected from the pretrained model for right encoder:{unexpected_keys}")

        logging.info(f"Encoders weights loaded from {ssl_path}")

        if self.shared_backbone:
            self.shared_backbone.load_state_dict(backbone_weights, strict=True)
            logging.info(f"Shared backbone weights loaded from {ssl_path}")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """Load model from checkpoint"""
        model = super().load_from_checkpoint(checkpoint_path, **kwargs)
        return model

    def forward_once(self, x, side):
        """Individual to side forward pass"""
        x = x.to(self.device)
        if side == "left":
            z = self.left_encoder(x)
        elif side == "right":
            z = self.right_encoder(x)
        else:
            raise NotImplementedError

        if self.shared_backbone:
            z = self.shared_backbone.forward(z)

        if self.projection_head:
            z = self.projection_head(z)

        return z

    def forward(self, x_left, x_right):
        z_left = self.forward_once(x_left, 'left')
        z_right = self.forward_once(x_right, 'right')

        return z_left, z_right

    def relational_step(self, similarity_matrix, batch_size, right_IDs, mode):
        """Training step for DRSL"""

        # Obtain loss
        loss, losses, labels = self.loss(similarity_matrix, right_IDs)

        # Get positive mask
        pos = torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        pos_sim = similarity_matrix[pos]

        # Extract negative pair examples (half a batch) for balanced dataset
        neg_sim = similarity_matrix[labels == 0]
        num_samples = batch_size
        indices = torch.randperm(batch_size)[:num_samples]
        neg_sim = neg_sim[indices]

        # Balanced metrics calculation
        similarity = torch.concat([pos_sim[:int(batch_size / 2)], neg_sim[int(batch_size / 2):]])
        match = torch.concat([torch.ones(floor(batch_size / 2)), torch.zeros(ceil(batch_size / 2))])
        match = match.to(self.device)

        pred_sim_labels = (similarity > self.hparams.similarity_threshold).float()
        correct = torch.sum((pred_sim_labels == match).float())
        accuracy = correct.detach().item() / match.size(0)

        # For validation and test, get and log topk metrics
        if (mode == 'validate' and self.hparams.val_batch_size == batch_size) or (mode == "test" and self.hparams.test_batch_size == batch_size):
            sim_matrix, label_matrix = self.uniques_sim_matrix(right_IDs, sim_matrix=similarity_matrix)
            top_ks, mean_pos = self.get_ks_acc(sim_matrix=sim_matrix, label_matrix=label_matrix, ks=[1, 3, 5, 10])
        else:
            if mode != "train":
                logging.info(f'Skipped smaller batch of {batch_size} as complete '
                             f'batch size is {self.hparams.val_batch_size if mode == 'validate'
                             else self.hparams.test_batch_size}')
            top_ks, mean_pos = None, None

        self.logging_everything(mode=mode, similarity=similarity, pos_sim=pos_sim, neg_sim=neg_sim,
                                loss=loss, mean_pos=mean_pos, top_ks=top_ks,
                                accuracy=accuracy)

        if mode == "test":
            return similarity
        else:
            return loss

    def non_relational_step(self, z_left, z_right, similarity, match, batch_size, right_IDs, mode=None):
        """Training for non-DRSL architecture"""
        if sum(match) == batch_size:
            # If all pairs are positive training
            # Reshuffling ids to get negative pairs
            right_IDs_lst = right_IDs.tolist()
            map_of_IDs = {key: idx for idx, key in enumerate(right_IDs_lst)}
            shuffled_IDs = right_IDs_lst.copy()
            while any(x == y for x, y in zip(right_IDs_lst, shuffled_IDs)):
                random.shuffle(shuffled_IDs)

            # Faking 50:50 positive:negative batch (to report average similarity and loss)
            match = torch.concat([torch.ones(floor(batch_size / 2)), torch.zeros(ceil(batch_size / 2))])
            match = match.to(self.device)

            # Calculating positive, negative and 50:50 similarities
            z_left_pos = z_left.clone()
            z_right_pos = z_right.clone()

            z_left_neg = z_left.clone()
            z_right_neg = torch.stack([z_right_pos[map_of_IDs[idx]] for idx in shuffled_IDs])

            z_right = torch.concat([z_right_pos[:int(batch_size / 2)], z_right_neg[int(batch_size / 2):]])

            pos_sim = similarity
            neg_sim = self.similarity(z_left, z_right_neg, dim=1, mode=mode)
            similarity = self.similarity(z_left, z_right, dim=1, mode=mode)

            # Make positive, negative match labels
            pos_match = torch.ones_like(pos_sim).to(self.device)
            neg_match = torch.zeros_like(neg_sim).to(self.device)
        else:
            # Else 50:50 pairs training
            # Get positive, negative pair indices
            pos_indices = (match == 1)
            neg_indices = (match != 1)

            # Get positive, negative feats
            z_right_pos = z_right[pos_indices]
            z_right_neg = z_right[neg_indices]

            z_left_pos = z_left[pos_indices]
            z_left_neg = z_left[neg_indices]

            # Get positive, negative similarities
            pos_sim = similarity[pos_indices]
            neg_sim = similarity[neg_indices]

            # Get positive, negative match labels
            pos_match = match[pos_indices]
            neg_match = match[neg_indices]

        # For validation and test, get and log topk metrics
        if (mode == 'validate' and self.hparams.val_batch_size == batch_size) or (
                mode == "test" and self.hparams.test_batch_size == batch_size):
            sim_matrix, label_matrix = self.uniques_sim_matrix(right_IDs, z_left=z_left_pos, z_right=z_right_pos)
            top_ks, mean_pos = self.get_ks_acc(sim_matrix=sim_matrix, label_matrix=label_matrix, ks=[1, 3, 5, 10])

        else:
            if mode != "train":
                logging.info(f'Skipped smaller batch of {batch_size} as complete '
                             f'batch size is {self.hparams.val_batch_size if mode == 'validate'
                             else self.hparams.test_batch_size}')
            top_ks, mean_pos = None, None

        # Get losses
        if self.hparams.loss_mode not in ["NT-Xent_In", "NT-Xent_Out", "NT-Xent_full", "Clip_loss", "Clip_loss2",
                                          "Relational_NN_loss"]:
            # Get 50:50, positive and negative loss
            loss = self.loss(z_left=z_left, z_right=z_right, match=match, similarity=similarity)
            pos_loss = self.loss(z_left=z_left_pos, z_right=z_right_pos, match=pos_match, similarity=pos_sim)
            neg_loss = self.loss(z_left=z_left_neg, z_right=z_right_neg, match=neg_match, similarity=neg_sim)
        else:
            # For "NT-Xent_In", "NT-Xent_Out", "NT-Xent_full", "Clip_loss","Relational_NN_loss"
            pos_loss, neg_loss = None, None
            try:
                loss = self.loss(z_left=z_left, z_right=z_right_pos, similarity=pos_sim, right_IDs=right_IDs)
            except RuntimeError as e:
                loss = 0

        # Get an overall accuracy
        pred_sim_labels = (similarity > self.hparams.similarity_threshold).float()
        correct = torch.sum((pred_sim_labels == match).float())
        accuracy = correct.detach().item() / match.size(0)

        temperature = self.loss.temperature.exp() if self.hparams.loss_mode == "Clip_loss" else None


        self.logging_everything(mode=mode, similarity=similarity, pos_sim=pos_sim, neg_sim=neg_sim,
                                loss=loss, pos_loss=pos_loss, neg_loss=neg_loss,
                                mean_pos=mean_pos, top_ks=top_ks,
                                accuracy=accuracy, temperature=temperature)

        if mode == "test":
            return similarity
        else:
            return loss

    def _step(self, batch, batch_idx, mode=None):
        x_left, x_right, match, right_IDs, _ = batch
        batch_size = len(right_IDs)
        match = match.to(self.device)
        right_IDs = right_IDs.to(self.device)

        # Forward pass
        z_left, z_right = self(x_left, x_right)

        # Calculate batch similarity
        similarity = self.similarity(z_left, z_right, dim=1)

        if self.hparams.similarity_mode == "Deep_Relational_Similarity":
           return self.relational_step(similarity_matrix=similarity, batch_size=batch_size, right_IDs=right_IDs, mode=mode)
        else:
           return self.non_relational_step(z_left=z_left, z_right=z_right, similarity=similarity, match=match, batch_size=batch_size, right_IDs=right_IDs, mode=mode)

    def uniques_sim_matrix(self, right_IDs, z_left=None, z_right=None, sim_matrix=None, return_uniques=False):
        # Create mask
        label_matrix = right_IDs == right_IDs.unsqueeze(1)

        # Remap indices from right_IDs to real indices
        uniques = []
        seen = []
        for idx, unique in enumerate(right_IDs):
            if unique.item() not in seen:
                seen.append(unique.item())
                uniques.append(idx)

        label_matrix = label_matrix[:, uniques]

        if z_left is not None and z_right is not None:
            # Remove duplicates
            unique_z_right = z_right[uniques]

            # Calculate similarity matrix
            sim_matrix = self.similarity(z_left.unsqueeze(1), unique_z_right.unsqueeze(0), dim=-1)
        elif sim_matrix is not None:
            # Remove duplicates
            sim_matrix = sim_matrix[:, uniques]
        else:
            raise AssertionError
        logging.info(f'Similarity matrix shape: {sim_matrix.shape}')

        if return_uniques:
            return sim_matrix, label_matrix, seen
        else:
            return sim_matrix, label_matrix

    def save_similarities(self, path, left_IDs, unique_right_IDs, sim_matrix, label_matrix):
        sim_matrix_np = sim_matrix.cpu().numpy()
        df = pd.DataFrame(
            sim_matrix_np,
            columns=unique_right_IDs,
            index=left_IDs.tolist(),
        )

        df = df.stack().reset_index()
        df.columns = ['PS', 'S2', 'sim']

        label_np = label_matrix.cpu().numpy()
        labels_df = pd.DataFrame(
            label_np,
            columns=unique_right_IDs,
            index=left_IDs.tolist(),
        )

        labels_df = labels_df.stack().reset_index()
        labels_df.columns = ['PS', 'S2', 'label']
        labels_df['label'] = labels_df['label'].astype(int)

        merged = df.merge(labels_df, on=['PS', 'S2'])

        # Save to CSV
        merged.to_csv(path, header=False, index=False)
        logging.info(f"Similarities saved to {path}")

    def get_ks_acc(self, sim_matrix: torch.tensor, label_matrix: torch.tensor, ks: list[int]):
        # Initialise dict of metrics
        top_ks = {k: [] for k in ks}
        positions = []

        for row, mask in zip(sim_matrix, label_matrix):
            labels = mask.int()

            # Sort based on similarity
            sorted_sim_idx = torch.argsort(row, descending=True)
            sorted_labels = labels[sorted_sim_idx]

            # Find position index
            pos_idx = torch.where(mask)[0][0]
            pos_position = (sorted_sim_idx == pos_idx).nonzero().item() + 1
            positions.append(pos_position)

            # Append if found in k
            for k in ks:
                top_ks[k].append(sorted_labels[:k].sum() > 0)

        # Get mean for how often real match was in k
        for k in ks:
            top_ks[k] = sum(top_ks[k]) / len(top_ks[k])

        mean_position = sum(positions) / len(positions)

        return top_ks, mean_position

    def logging_everything(self, mode, similarity=None, pos_sim=None, neg_sim=None, loss=None, pos_loss=None,neg_loss=None, temperature=None, accuracy=None, mean_pos=None, top_ks=None):
        # Log Similarities
        if similarity is not None:
            self.log(f'{mode}_similarity', similarity.mean().detach().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if pos_sim is not None and neg_sim is not None:
            self.log(f'{mode}_pos_similarity', pos_sim.mean().detach().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'{mode}_neg_similarity', neg_sim.mean().detach().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Log losses
        if loss:
            self.log(f'{mode}_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if pos_loss and neg_loss:
            self.log(f'{mode}_pos_loss', pos_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'{mode}_neg_loss', neg_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Log temperature for clip loss
        if temperature:
            self.log("temperature", temperature, on_step=False, on_epoch=True, prog_bar=False,
                     logger=True)

        # Log accuracy
        if accuracy:
            self.log(f'{mode}_acc', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if mean_pos:
            self.log(f'{mode}_mean_pos', mean_pos, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if top_ks is not None:
            for k in top_ks:
                self.log(f'{mode}_top_{k}_acc', top_ks[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'validate')

    def test_step(self, batch, batch_idx, ):
        return self._step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, return_features=False, save_path=None):
        x_left, x_right, match, right_IDs, left_IDs = batch
        del batch
        batch_size = len(right_IDs)
        match = match.to(self.device)
        right_IDs = right_IDs.to(self.device)

        # Forward pass
        with torch.no_grad():
            z_left, z_right = self(x_left, x_right)
            del x_left, x_right

        if save_path is None and return_features:
            if self.hparams.similarity_mode == "Deep_Relational_Similarity":
                similarity = self.similarity(z_left, z_right, dim=-1)
            else:
                similarity = self.similarity(z_left, z_right, dim=1)

            return z_left.detach().cpu(), z_right.detach().cpu(), left_IDs, right_IDs, similarity, match
        elif save_path and self.hparams.test_batch_size == batch_size:
            if self.hparams.similarity_mode == "Deep_Relational_Similarity":
                similarity = self.similarity(z_left, z_right, dim=1)
                sim_matrix, label_matrix, unique_right_IDs = self.uniques_sim_matrix(right_IDs, sim_matrix=similarity, return_uniques=True)
                del similarity
            else:
                sim_matrix, label_matrix, unique_right_IDs = self.uniques_sim_matrix(right_IDs, z_left=z_left, z_right=z_right, return_uniques=True)

            del z_left, z_right, match, right_IDs

            self.save_similarities(path=save_path, left_IDs=left_IDs, unique_right_IDs=unique_right_IDs, sim_matrix=sim_matrix, label_matrix=label_matrix)

            del sim_matrix, label_matrix, unique_right_IDs, left_IDs
        else:
            raise AssertionError("Batch size should be equal to test dataset length for saving top-k metrics")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr / 50)
        return [optimizer], [lr_scheduler]