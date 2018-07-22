import os
import nltk
import pickle
import json
import argparse
from collections import Counter
import numpy as np

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_glove_voc(threshold, vocab):

    data_path = '/corpus/glove/pretrained_vector/english/glove.42B.300d.{}'
    count = 0
    weight_matrix = np.random.uniform(-0.5, 0.5, size=(threshold,300))

    with open(data_path.format('txt'),'r', encoding='utf8') as f:
        for line in f:
            l = line.strip().split()
            word = l[0]
            if vocab(word) != 3:
                weight_matrix[vocab(word),:] = np.asarray(list(map(float, l[1:])))

            count += 1
    print(weight_matrix.shape)
    return weight_matrix

def build_vocab(text, threshold):
    """Build a simple vocabulary wrapper."""
    train = json.load(open(text[0], 'r'))
    valid = json.load(open(text[1], 'r'))
    test = json.load(open(text[2], 'r'))

    counter = Counter()

    # update counter
    counter_update(train, counter)
    counter_update(valid, counter)
    counter_update(test, counter)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def counter_update(annotations, counter):
    for i , data in enumerate(annotations['dialogs']):
        caption = data['caption']
        dialog = data['dialog']
        summary = data['summary']

        for qa in dialog:
            qa_pair = qa['question']+' '+qa['answer']
            description = caption+' '+summary+' '+qa_pair
            tokens = nltk.tokenize.word_tokenize(description)
            counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(annotations['dialogs'])))


def main(args):
    if not os.path.exists(args.vocab_dir):
        os.makedirs(args.vocab_dir)
        print("Make Data Directory")
    vocab = build_vocab(text=[args.train_path, args.val_path, args.test_path],
                        threshold=args.threshold)
    W = build_glove_voc(len(vocab), vocab)
    vocab_path = os.path.join(args.vocab_dir, 'vocab.pkl')
    weight_path = os.path.join(args.vocab_dir, 'pretrained_embedding.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(weight_path, 'wb') as f:
        pickle.dump(W, f)

    print("\nTotal vocabulary size: %d" %len(vocab))
    print("Total word embedding size: %d" % len(W))
    print("Saved the vocabulary wrapper to '%s'\n" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default='/home/iris1006/avsd/data/annotations/train_set.json',
                        help='path for train annotation file')
    parser.add_argument('--val_path', type=str,
                        default='/home/iris1006/avsd/data/annotations/valid_set.json',
                        help='path for validation annotation file')
    parser.add_argument('--test_path', type=str,
                        default='/home/iris1006/avsd/data/annotations/test_set.json',
                        help='path for test annotation file')
    parser.add_argument('--vocab_dir', type=str, default='/home/iris1006/avsd/v_wrapper/',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=3,
                        help='minimum word count threshold')

    args = parser.parse_args()
    main(args)