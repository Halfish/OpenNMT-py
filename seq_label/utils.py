import random
import time
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


torch.manual_seed(1)
random.seed(1)

# speical symbol for `class vocab`
PAD_SYMBOL = '<pad>'
UNK_SYMBOL = '<unk>'

# special tag for CRF model
START_TAG = '<start>'
STOP_TAG = '<stop>'
tag_to_ix = {'0':0, '1':1, START_TAG:2, STOP_TAG:3}


def log_sum_exp(vec, batch_size):
    max_score, _ = torch.max(vec, dim=1)
    max_score_broadcast = max_score.view(
        batch_size, -1).expand(batch_size, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))


class Vocab(object):
    def __init__(self, min_count=100):
        self.min_count = min_count
        self.special_words = [PAD_SYMBOL, UNK_SYMBOL]
        self.wordcount = {}
        self._word2id = {}      # shoud not be accessed publicly
        self._id2word = []      # shoud not be accessed publicly
        self.closed = False     # read self.close() to see why we need this flag

    def update(self, words):
        assert not self.closed, 'already closed, stop updating'
        if not isinstance(words, list):
            words = list(words) # words should be list of str
        for word in words:
            if not word in self.wordcount:
                self.wordcount[word] = 1
            else:
                self.wordcount[word] += 1

    def close(self):
        # filter low-frequency words and close update
        self._id2word = self.special_words + [word for word, count in \
                self.wordcount.items() if count >= self.min_count]
        self._word2id = {word:i for i, word in enumerate(self._id2word)}
        self.closed = True

    def build_from_file(self, filename):
        for line in open(filename, 'r'):
            words = [word[:-2] for word in line.strip().split(' ')]
            self.update(words)
        self.close()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._id2word[key]
        elif isinstance(key, str):
            key = key if key in self._word2id else UNK_SYMBOL
            return self._word2id[key]
        else:
            raise KeyError("invalid key! Not str or int.")

    def __len__(self):
        return len(self._id2word)

    def save(self, filename):
        print('save vocab as', filename)
        with open(filename, 'wb') as fwrite:
            pickle.dump(self, fwrite)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as fread:
            return pickle.load(fread)


class Dataset(object):
    def __init__(self, filename, args, vocab=None):
        self.dataset = [line.strip() for line in open(filename, 'r')]
        self.args = args
        self.vocab = vocab if vocab else self._init_vocab()

    def _init_vocab(self):
        vocab = Vocab()
        for data in self.dataset:
            words = [word[:-2] for word in data.split(' ')]
            vocab.update(words)
        vocab.close()
        return vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        origin_data = self.dataset[index].replace('||', '@|') # fix bug
        data = [w.split('|') for w in origin_data.split(' ')]
        sent, tag = list(zip(*data))
        sent = [self.vocab[w] for w in sent]
        tag = [int(t) for t in tag]
        return sent, tag

    def __iter__(self):
        count = 0
        batch_data = []
        for i in range(self.__len__()):
            batch_data.append(self[i])
            count = count + 1
            if count % self.args.batch_size == 0:
                yield self.padding_batch(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield self.padding_batch(batch_data)

    def next_batch(self):
        batch_data = []
        for i in range(self.args.batch_size):
            index = random.randint(0, len(self.dataset) - 1)
            batch_data.append(self.__getitem__(index))
        return self.padding_batch(batch_data)

    def padding_batch(self, batch_data):
        batch_sents, batch_tags = list(zip(*batch_data)) # split batch_data
        batch_lengths = [len(sent) for sent in batch_sents]

        max_len = max(batch_lengths)
        padded_sents = [self._padding(sent, max_len) for sent in batch_sents]
        padded_tags = [self._padding(tag, max_len, 0) for tag in batch_tags]

        sentences = Variable(torch.LongTensor(padded_sents))
        tags = Variable(torch.LongTensor(padded_tags))
        lengths = Variable(torch.LongTensor(batch_lengths))

        if self.args.use_gpu:
            sentences, tags, lengths = sentences.cuda(), tags.cuda(), lengths.cuda()
        return sentences, tags, lengths

    def _padding(self, array, max_len, padding_id=None):
        assert max_len >= len(array)
        padding_id = padding_id if padding_id else self.vocab[PAD_SYMBOL]
        return list(array) + [padding_id for i in range(max_len - len(array))] 


class SeqLabelModel(nn.Module):
    '''
    sequence labelling model with LSTM-CRF
    '''
    def __init__(self, vocab, args):
        super(SeqLabelModel, self).__init__()
        self.vocab = vocab
        self.args = args
        '''
        self.weight = Variable(torch.FloatTensor([1, 10, 1, 1]))
        self.weight = self.weight / self.weight.mean()
        assert len(weight) == self.tag_dim
        if self.self.args.use_gpu:
            self.weight = self.weight.cuda()
        '''
        self.tag_dim = self.args.tag_dim + 2 if self.args.use_crf else self.args.tag_dim
        self.num_directions = 2 if self.args.bidirectional else 1
        self.embedding = nn.Embedding(
                len(vocab), self.args.embedding_dim, padding_idx=vocab[PAD_SYMBOL])
        self.lstm = nn.LSTM(self.args.embedding_dim, self.args.hidden_dim, 
                            num_layers=self.args.num_layers, 
                            bidirectional=self.args.bidirectional, 
                            batch_first=False)
        self.hidden2tag = nn.Linear(
                    self.args.hidden_dim * self.num_directions, self.tag_dim)

        if self.args.use_crf:
            # transition matrix for CRF, Entry i,j is the score of 
            # transitioning to i from j
            self.transitions = nn.Parameter(torch.randn(self.tag_dim, self.tag_dim))
            # constraint we never transfer to the start tag or from the stop tag
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 8]))

    def _init_hidden(self, batch_size=None):
        batch_size = batch_size if batch_size else self.args.batch_size
        h_0 = Variable(torch.randn(self.args.num_layers * self.num_directions, 
                                    batch_size, self.args.hidden_dim))
        c_0 = Variable(torch.randn(self.args.num_layers * self.num_directions,
                                    batch_size, self.args.hidden_dim))
        if self.args.use_gpu:
            h_0, c_0 = h_0.cuda(), c_0.cuda()

        return h_0, c_0

    def get_lstm_features(self, sentence):
        h, c = self._init_hidden(batch_size=sentence.size(0))
        # embeds is (seq_len, batch_size, embedding_dim)
        embeds = self.embedding(sentence).transpose(0, 1) 
        # lstm_out is (seq_len, batch_size, hidden_dim * 2)
        lstm_out, (h, c) = self.lstm(embeds, (h, c))
        # lstm_feats is (seq_len, batch_size, tagset_size)
        lstm_feats = self.hidden2tag(lstm_out) 
        return lstm_feats

    def _forward_alg(self, feats):
        '''Do the forward algorithm to compute the partition function
        '''
        init_alphas = torch.Tensor(self.args.batch_size, self.tag_dim).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[:, tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = Variable(init_alphas)
        if self.args.use_gpu:
            forward_var = forward_var.cuda()

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tag_dim):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].unsqueeze(1).expand(-1, self.tag_dim)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag]
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(self.tag_dim, self.args.batch_size)
            forward_var = forward_var.transpose(0, 1).contiguous()
        terminal_var = forward_var + self.transitions[tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        '''
        Args:
            feats: FloatTensor  (seq_len, batch_size, tag_dim)
            tags: LongTensor    (batch_size, seq_len)
        '''
        #tag_weight = self.weight[tags.view(-1)].view_as(tags)
        #stop_tag_weight = self.weight[tag_to_ix[STOP_TAG]].expand(self.args.batch_size)

        # Gives the score of a provided tag sequence
        start_ids = torch.LongTensor([tag_to_ix[START_TAG]] * self.args.batch_size)
        start_ids = Variable(start_ids.view(self.args.batch_size, -1))
        if self.args.use_gpu:
            start_ids = start_ids.cuda()
        tags = torch.cat([start_ids, tags], dim=1)

        trans_score = self.transitions[tags[:, 1:], tags[:, :-1]] # (batch_size, seq_len)
        feat_score = torch.gather(feats, 2, tags[:, 1:].transpose(0, 1).unsqueeze(-1)
                        ).squeeze(-1).transpose(0, 1) # (batch_size, seq_len)
        score = torch.sum(trans_score + feat_score, dim=1)
        score = score + self.transitions[tag_to_ix[STOP_TAG], tags[:, -1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(self.args.batch_size, self.tag_dim).fill_(-10000.)
        init_vvars[:, tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        if self.args.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tag_dim):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_var, best_tag_id = torch.max(next_tag_var, 1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(best_var)
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = torch.stack(viterbivars_t, 1) + feat
            backpointers.append(torch.stack(bptrs_t, 1))

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[tag_to_ix[STOP_TAG]]
        path_score, best_tag_id = torch.max(terminal_var, 1)

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = torch.cat(
                    [bptrs_t[i, id_t] for i, id_t in enumerate(best_tag_id)])
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert (start == tag_to_ix[START_TAG]).all()  # Sanity check
        best_path.reverse()
        best_path = torch.stack(best_path, 1)
        return path_score, best_path

    @staticmethod
    def binary_accuracy(outputs, targets):
        outputs = outputs.data.contiguous().view(-1).byte()
        targets = targets.data.contiguous().view(-1).byte()
        assert (outputs.numel() == targets.numel()), 'size do not match'
        true_accu = (outputs & targets).float().sum() / targets.float().sum()
        false_accu = (~(outputs | targets)).float().sum() / (~targets).float().sum()
        return true_accu, false_accu

    def neg_log_likelihood(self, sentence, tags):
        '''training and validation for bi-lstm-crf model
        '''
        feats = self.get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        loss = (forward_score - gold_score).mean()
        _, tag_pred = self._viterbi_decode(feats)
        true_accu, false_accu = SeqLabelModel.binary_accuracy(tag_pred, tags)
        return loss, true_accu, false_accu

    def lstm_forward(self, sentence, tags):
        '''training and validation for lstm model
        '''
        feats = self.get_lstm_features(sentence) # (seq_len, batch_size, tag_dim)
        scores = feats.transpose(0, 1).contiguous().view(-1, feats.size(2))
        tags = tags.view(-1)
        loss = self.loss_fn(scores, tags)
        _, indices = torch.max(scores, -1)
        true_accu, false_accu = SeqLabelModel.binary_accuracy(indices, tags)
        return loss, true_accu, false_accu

    def forward(self, sentence):
        '''
        prediction for lstm model or bi-lstm-crf model
        '''
        # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self.get_lstm_features(sentence)
        if not self.args.use_crf:
            return lstm_feats
        else:
            # Find the best path, given the features.
            score, tag_seq = self._viterbi_decode(lstm_feats)
            return score, tag_seq


class Trainer(object):
    def __init__(self, model, args, train_dataset, valid_dataset, writer):
        self.model = model
        self.args = args
        self.start_epoch = 0
        self.writer = writer # write data to tensorboardX
        self.learning_rate = self.args.learning_rate
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        if self.args.use_gpu:
            self.model = self.model.cuda()

    def train(self):
        '''random choose batch data and training it on the model
        '''
        self.model.train()
        sentences, tags, lengths = self.train_dataset.next_batch()
        if self.args.use_crf:
            loss, true_accu, false_accu = self.model.neg_log_likelihood(sentences, tags)
        else:
            loss, true_accu, false_accu = self.model.lstm_forward(sentences, tags)
        self.optimizer.zero_grad()
        loss.backward() 
        total_norm = torch.nn.utils.clip_grad_norm(
                                self.model.lstm.parameters(), self.args.max_clip_norm)
        self.optimizer.step()
        return loss.data, true_accu, false_accu

    def validate(self, epoch):
        '''different from above function, 
            this function loop over the whole validation dataset
        '''
        self.model.eval()
        valid_loss = []
        for batchid in range(len(self.valid_dataset) // self.args.batch_size):
            sentences, tags, lengths = self.valid_dataset.next_batch()
            if self.args.use_crf:
                loss, true_accu, false_accu = self.model.neg_log_likelihood(sentences, tags)
            else:
                loss, true_accu, false_accu = self.model.lstm_forward(sentences, tags)
            print(('\tepoch %d, batchid=%d, valid_loss=%.5f,' 
                    ' true_accu=%.5f, false_accu=%.5f') 
                    % (epoch, batchid, loss.data[0], true_accu, false_accu))
            valid_loss.append(loss.data)
        valid_loss = torch.cat(valid_loss).mean()
        return valid_loss, true_accu, false_accu

    def loop_epoches(self):
        print('start to train')
        for epoch in range(self.start_epoch, self.args.total_epoches):
            train_loss = []
            total_batch = len(self.train_dataset) // self.args.batch_size
            for batchid in range(total_batch):
                # training
                batch_loss, true_accu, false_accu = self.train()
                actual_batchid = epoch * total_batch + batchid
                self.writer.add_scalar('data/train_loss', batch_loss[0], actual_batchid)
                self.writer.add_scalar('data/train_true_accu', true_accu, actual_batchid)
                self.writer.add_scalar('data/train_false_accu', false_accu, actual_batchid)
                print(('epoch %d, batchid=%d/%d, '
                    'train_loss=%.5f, true_accu=%.5f, false_accu=%.5f') 
                    % (epoch, batchid, total_batch, batch_loss[0], true_accu, false_accu))
                train_loss.append(batch_loss)
                # validate
                if (batchid + 1) % self.args.valid_freq == 0:
                    valid_loss, true_accu, false_accu = self.validate(epoch)
                    print(('\tepoch %d, batchid=%d/%d, valid_loss=%.5f,'
                        ' true_accu=%.5f, false_accu=%.5f') 
                        % (epoch, batchid, total_batch, valid_loss, true_accu, false_accu))
                    self.writer.add_scalar('data/valid_loss', valid_loss, actual_batchid)
                    self.writer.add_scalar('data/v_true_accu', true_accu, actual_batchid)
                    self.writer.add_scalar('data/v_false_accu', false_accu, actual_batchid)
                    self.writer.add_embedding(self.model.embedding.weight, tag='words',
                            metadata=self.train_dataset.vocab._id2word, global_step=epoch)
                # learning rate decay
                if epoch >= self.args.lr_start_decay and batchid % self.args.lr_decay_freq == 0:
                    self.learning_rate = self.learning_rate / 2
                    self.optimizer.param_groups[0]['lr'] = self.learning_rate
                    print('decay lr to %f' % (self.learning_rate))
                self.writer.add_scalar('data/learning', self.learning_rate, actual_batchid)
            valid_loss, true_accu, false_accu = self.validate(epoch)
            train_loss = torch.cat(train_loss)[-10:].mean()
            self.save_checkpoint(epoch, train_loss, valid_loss, true_accu, false_accu)

    def save_checkpoint(self, epoch, train_loss, valid_loss, true_accu, false_accu):
        self.checkpoint = {
                'model': self.model.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'true_accu': true_accu,
                'false_accu': false_accu,
                'optimizer': self.optimizer,
                'epoch': epoch,
                'args': self.args
        }
        print(('saving model, epoch = %d, train_loss = %.5f,'
                'valid_loss = %.5f') % (epoch, train_loss, valid_loss))
        torch.save(self.checkpoint, 'checkpoint_%d.pt' % (epoch))

    def load_checkpoint(self, chpt_name):
        self.checkpoint = torch.load(chpt_name)
        self.model.load_state_dict(self.checkpoint['model'])
        self.start_epoch = self.checkpoint['epoch'] + 1
        self.optimizer = self.checkpoint['optimizer']
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        print("\n%s loaded, learning_rate = %.6f" % (chpt_name, self.learning_rate))
        print(('train_loss = %.5f, valid_loss = %.5f '
                'true_accu = %.5f, false_accu = %.5f')
                % (self.checkpoint['train_loss'], self.checkpoint['valid_loss'], 
                    self.checkpoint['true_accu'], self.checkpoint['false_accu']))
