import torch
import torch.nn as nn
import torch.nn.functional as F


# Simplified implemented from https://github.com/violatingcp/codec/blob/main/losses.py
class SimCLRLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = features #torch.cat(torch.unbind(features, dim=1), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(1, contrast_count)
        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
        #     0
        # )
        # mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits)  #* logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss


class VICRegLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        repr_loss = F.mse_loss(x, y)

        x_mu = x.mean(dim=0)
        x_std = x.std(dim=0) + 1e-2
        y_mu = y.mean(dim=0)
        y_std = y.std(dim=0) + 1e-2

        x = (x - x_mu)/x_std
        y = (y - y_mu)/y_std

        N = x.size(0)
        D = x.size(-1)

        std_loss = torch.mean(F.relu(1 - x_std, inplace=False)) / 2
        std_loss += torch.mean(F.relu(1 - y_std, inplace=False)) / 2

        cov_x = (x.transpose(1, 2).contiguous() @ x) / (N - 1)
        cov_y = (y.transpose(1, 2).contiguous() @ y) / (N - 1)

        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)

        return repr_loss + cov_loss + std_loss

    def off_diagonal(self, x):
        num_batch, n, m = x.shape
        assert n == m
        # All off diagonal elements from complete batch flattened
        return x.flatten(start_dim=1)[...,:-1].view(num_batch, n - 1, n + 1)[...,1:].flatten()


# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import activations

# # Contrastive Loss Function
# def SimCLRLoss(features, labels, temperature = 0.07):
#     '''
#     Computes SimCLRLoss as defined in https://arxiv.org/pdf/2004.11362.pdf
#     '''
#     batch_size = features.shape[0]
#     if (features.shape[0] != labels.shape[0]):
#         raise ValueError('Error in SIMCLRLOSS: Number of labels does not match number of features')

#     # Generates mask indicating what samples are considered pos/neg
#     positive_mask = tf.equal(labels, tf.transpose(labels))
#     negative_mask = tf.logical_not(positive_mask)
#     positive_mask = tf.cast(positive_mask, dtype=tf.float32)
#     negative_mask = tf.cast(negative_mask, dtype=tf.float32)

#     # Computes dp between pairs
#     logits = tf.linalg.matmul(features, features, transpose_b=True)
#     temperature = tf.cast(temperature, tf.float32)
#     logits = logits / temperature

#     # Subtract largest |logits| elt for numerical stability
#     # Simply for numerical precision -> stop gradient
#     max_logit = tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
#     logits = logits - max_logit

#     exp_logits = tf.exp(logits)
#     num_positives_per_row = tf.reduce_sum(positive_mask, axis=1)

#     denominator = tf.reduce_sum(exp_logits * negative_mask, axis = 1, keepdims=True)
#     denominator += tf.reduce_sum(exp_logits * positive_mask, axis = 1, keepdims=True)

#     # Compute L OUTSIDE -> defined in eq 2 of paper
#     log_probs = (logits - tf.math.log(denominator)) * positive_mask
#     log_probs = tf.reduce_sum(log_probs, axis=1)
#     log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
#     loss = -log_probs * temperature
#     loss = tf.reduce_mean(loss, axis=0)
#     return loss

# def VicRegLoss(x, y):
#     '''
#     Computes VicRegLoss as implemented by Deep
#     '''
#     # Finds float values of batch_size, dimension
#     N = tf.cast(tf.shape(x)[0], dtype=tf.float32)
#     D = tf.cast(tf.shape(x)[1], dtype=tf.float32)

#     # Calculate invariance term -> mse between x, y pairs
#     invariance_loss = keras.losses.mean_squared_error(x, y)

#     # Calculate variance_loss -> push std over batch of each variable towards γ=1
#     x_mu = tf.reduce_mean(x, axis=0)
#     y_mu = tf.reduce_mean(y, axis=0)
#     x_std = tf.sqrt(tf.math.reduce_variance(x, axis=0) + 0.0001)
#     y_std = tf.sqrt(tf.math.reduce_variance(y, axis=0) + 0.0001) # 0.0001 term for numberical stability
#     varaince_loss = tf.reduce_mean(tf.maximum(0.0, 1-x_std))/2 + tf.reduce_mean(tf.maximum(0.0, 1-y_std))/2

#     x = (x-x_mu)/x_std
#     y = (y-y_mu)/y_std

#     # Calculate covariance_loss -> pushes cov between variables to 0, prevent info collapse
#     cov_x = tf.matmul(tf.transpose(x), x) / (N-1)
#     cov_y = tf.matmul(tf.transpose(y), y) / (N-1)

#     # Covariance only relevant for off-diagonal elements of x,y
#     off_diag_mask = tf.math.logical_not(tf.eye(N, dtype=tf.bool))
#     x_off_diag = tf.reshape(x[off_diag_mask], (N-1, N+1))[:, 1:]
#     y_off_diag = tf.reshape(y[off_diag_mask], (N-1, N+1))[:, 1:]
#     covariance_loss = tf.reduce_sum(tf.pow(x_off_diag), 2) / D + tf.reduce_sum(tf.pow(y_off_diag), 2) / D

#     # Sums respective loss terms and returns output
#     return invariance_loss + variance_loss + covariance_loss

# def mse_loss(inputs, outputs):
#     return tf.math.reduce_mean(tf.math.square(outputs-inputs), axis=-1)

# def reco_loss(inputs, outputs):
#     # reshape inputs and outputs to manipulate p_t, n, phi values
#     inputs = tf.reshape(inputs, (-1,19,3,1))
#     outputs = tf.reshape(outputs, (-1,19,3,1))

#     # impose physical constraints on phi+eta for reconstruction
#     tanh_outputs = tf.math.tanh(outputs)
#     outputs_phi = math.pi*tanh_outputs
#     outputs_eta_egamma = 3.0*tanh_outputs
#     outputs_eta_muons = 2.1*tanh_outputs
#     outputs_eta_jets = 4.0*tanh_outputs
#     outputs_eta = tf.concat(
#         [outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:],
#          outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
#     outputs = tf.concat([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)

#     # zero features -> no particles. Disgard loss @ those values through masking
#     inputs = tf.squeeze(inputs, -1)
#     mask = tf.math.not_equal(inputs, 0)
#     mask = tf.cast(mask, tf.float32)
#     outputs = outputs * mask

#     # returns mse loss between reconstruction and real values
#     reconstruction_loss = mse_loss(tf.reshape(inputs, (-1,57)), tf.reshape(outputs, (-1,57)))
#     return tf.math.reduce_mean(reconstruction_loss)