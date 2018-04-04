from utils import Vocab, Dataset, SeqLabelModel
from opts import set_arguments
import torch.nn.functional as F
import argparse
import torch


parser = argparse.ArgumentParser()
set_arguments(parser)
args = parser.parse_args()
print(args)

vocab = Vocab.load('train.vocab')
print(vocab)

dataset = Dataset(filename='../weibo/src-valid.keywords.txt', args=args, vocab=vocab)
print(dataset.vocab)
print(dataset)

checkpoint = torch.load('checkpoint_6.pt')
print(checkpoint.keys())

model = SeqLabelModel(vocab, args)
model.load_state_dict(checkpoint['model'])
print(model)

model = model.eval().cuda()

keywords = []
count = -1
for sents, tags, lengths in dataset:
    lstm_features = model.get_lstm_features(sents)
    scores = F.softmax(lstm_features, -1).transpose(0, 1).contiguous().data[:, :, 1]
    # scores (batch_size, seq_len)
    lengths = lengths.data
    if count > 100000:
        break
    for i, score in enumerate(scores):
        count += 1
        if count % 1000 == 0:
            print('count = ', count)
        score = score[0:lengths[i]]
        _, indices = torch.sort(score, 0, descending=True)
        keyword = []
        line = [w[:-2] for w in dataset.dataset[count].split(' ')]
        for j in range(lengths[i]):
            w = line[indices[j]]
            if w not in keyword:
                keyword.append(w)
        keywords.append(keyword)
with open('valid.keywords.txt', 'w') as fwrite:
    fwrite.write('\n'.join([' '.join(s) for s in keywords]))
