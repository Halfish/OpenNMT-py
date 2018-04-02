"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import onmt
import onmt.io


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, tag_score, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, tag_score, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, tag_score, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, tag_score, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, tag_score, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note harding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          tag_score(FloatTensor) :
                output of tag score : [src_len x batch_size x tag_dim]
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, tag_score, output, range_, attns)

        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)

            loss.div(normalization).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, key_loss, gamma, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss[0], key_loss, gamma, non_padding.sum(), num_correct)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)

        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
            self.tag_criterion = nn.CrossEntropyLoss(
                        weight=torch.FloatTensor([1, 8]), reduce=False)
        self.confidence = 1.0 - label_smoothing
        self.count = 0

    def _make_shard_state(self, batch, tag_score, output, range_, attns=None):
        return {
            "tag_score": tag_score,
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "attns": attns['std'],
        }

    def keyphrase_coverage_loss(self, output, target, size_average=False):
        ''' output and target should all be both (batch_size, seq_len)
            Also notice that tmp may contains zeros, which means 
                there is no keyphrase, so the key_loss should be zero
        '''
        #tmp = torch.sum(output * target, dim=1) 
        #loss = 1.0 / (tmp + 1e-8)
        #loss = tmp.gt(0).float() * loss # mask output invalid loss
        loss = target * (1 / (torch.min(output, torch.ones_like(output)) + 1e-9) - 1)
        if size_average:
            return loss.mean()
        else:
            return loss.sum()

    def _compute_loss(self, batch, tag_score, output, target, attns):
        '''
            tag_score (src_len, batch_size, tag_dim)
            output (tgt_len, batch_size, hidden_dim)
            attns (tgt_len, batch_size, src_len)
        '''
        scores = self.generator(self._bottle(output))

        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        pred_loss = self.criterion(scores, gtruth)

        # keyphrase predict loss
        tag_weight = onmt.Utils.sequence_mask(batch.src[1]) # (batch_size, src_len)
        tag_weight = Variable(tag_weight).float().transpose(0, 1).contiguous()
        '''
        tag_target = batch.src_feat_0 - 2
        tag_loss = self.tag_criterion(tag_score.view(-1, 2), tag_target.view(-1))
        tag_loss = torch.mean(tag_weight * tag_loss.view_as(tag_weight))
        '''

        # soft keyphrase coverage loss
        '''
        tag_pred = F.softmax(tag_score, -1)[:, :, 1] # (src_len, batch_size)
        tag_attn = attns.sum(0).transpose(0, 1) # (src_len, batch_size)
        tag_cover_loss = tag_weight * torch.max(
                        tag_pred - tag_attn, torch.zeros_like(tag_attn))
        tag_cover_loss = tag_cover_loss.sum()
	'''
        # keyphrase coverage loss
        key_output = attns.sum(0)
        key_target = (tag_weight * (batch.src_feat_0 - 2).float()).transpose(0, 1)
        key_cover_loss = self.keyphrase_coverage_loss(key_output, key_target)

        # total loss
        if self.count < 300000:
            gamma = 0
        else:
            gamma = min(self.count / 100000 - 3, 1)
        self.count += 1
        loss = pred_loss + gamma * key_cover_loss

        if self.confidence < 1:
            loss_data = - likelihood.sum(0)
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, 
                key_cover_loss.data[0], gamma, scores.data, target.view(-1).data)

        return loss, stats


def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
