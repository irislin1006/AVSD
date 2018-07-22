from __future__ import absolute_import, division, print_function

import numpy as np
import os
import argparse
import pickle
import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
import data_loader
from build_vocab import Vocabulary


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='assign gpu')
    parser.add_argument('--sample', type=int, default=-1, help='No samle: less than zero, sample can further be the number of sample beam sizes')
    parser.add_argument('--output_path', type=str, default='/home/iris1006/avsd/output', help='Records for sample')
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='/home/iris1006/avsd/v_wrapper/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--glove_path', type=str, default='/home/iris1006/avsd/v_wrapper/pretrained_embedding.pkl',
                        help='path for pretained word embedding')
    parser.add_argument('--log_step', type=int, default=30, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=50, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--modality_dim', type=int, default=2048, help='step size for prining log info')
    parser.add_argument('--embed_size', type=int, default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--maxlen', type=int, default=20)
    return parser.parse_args()

def build_model(args, pretrained_embedding, v_dim, a_dim, l_dim):
    from model import BaseModel
    model = BaseModel(args, pretrained_embedding, v_dim, a_dim, l_dim)
    return model.cuda()

def grad_or_not(model):
    for name,p in model.named_parameters():
        if p.requires_grad:
            print('yes grad',name)
        else:
            print('no grad', name)

def train(args):
    # Load vocabulary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(args.glove_path, 'rb') as f:
        pretrained_embedding = pickle.load(f)

    trainloader, valloader, testloader = data_loader.prepare_all_loaders(batch_size=args.batch_size, vocab=vocab)
    # Build models
    v_dim, a_dim, l_dim = args.modality_dim, args.modality_dim, (args.hidden_size)*2
    model = build_model(args, pretrained_embedding, v_dim, a_dim, l_dim)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # grad_or_not(model)

    # Train the models
    best_val_acc = 0
    total_step = len(trainloader)
    for epoch in range(args.num_epochs):
        loss_record = []
        for i, (images, audios, caption, dialoges, lengths) in enumerate(trainloader):
            # Assure in training mode
            model.train()
            # print('IN DATALOADER')
            # grad_or_not(model)
            # print('model.train() image:',images.requires_grad)
            # Set mini-batch dataset
            images = images.cuda()
            # print('images = images.cuda():', images.requires_grad)
            audios = audios.cuda()
            captions = caption.cuda()
            dialoges = dialoges.cuda()
            _lengths = lengths['captions']
            targets = pack_padded_sequence(captions, _lengths, batch_first=True)[0].cuda()
            # print('targets = pack_padded_sequence:', targets.requires_grad)

            # Forward, backward and optimize
            outputs, BS, features = model(images, audios, dialoges, captions, _lengths)
            # print('Forward, backward and optimize')
            # grad_or_not(model)
            # print('outputs, BS, features = model', outputs.requires_grad)
            # print('outputs, BS, features = Image', images.requires_grad)
            loss = criterion(outputs.view(-1, len(vocab)), targets)
            loss_record.append(loss.item())
            # print('loss = criterion:', loss.requires_grad)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, AVG Loss: {:.4f}\n'
                      .format(epoch+1, args.num_epochs, \
                              i+1, total_step, \
                              loss.item(), \
                              np.exp(loss.item()), \
                              sum(loss_record)/len(loss_record)))
                loss_record = []

            # Make evaluation on validation set, and save model
            if (i + 1) % args.save_step == 0:
                # Make sure in the evaluation mode
                model.eval()
                current_score = val(epoch, model, valloader, criterion)

                if best_val_acc != 0 or current_score > best_val_acc:
                    best_val_acc = current_score
                    print('\nSaving Model...\n')
                    torch.save(model.state_dict(), os.path.join(
                        args.model_path, 'model-{}-{}-valloss-[{}].ckpt'.format(epoch + 1, i + 1, current_score)))

                # Print sample
                generate_sample(args.output_path, epoch, i, model, features, captions, _lengths, outputs, BS, vocab)


def val(epoch, model, dataloader, criterion):
    print('Evaluating validation preformance...')
    loss_sum = 0
    # print(len(dataloader))
    for i, (images, audios, caption, dialoges, lengths) in enumerate(dataloader):
        # print(i, images.size(), audios.size(), dialoges.size(), caption.size())
        images = images.cuda()
        audios = audios.cuda()
        _lengths = lengths['captions']
        captions = caption.cuda()
        dialoges = dialoges.cuda()
        targets = pack_padded_sequence(captions, _lengths, batch_first=True)[0].cuda()

        pred, _ , features = model(images, audios, dialoges, captions, _lengths)
        # print('P:',type(pred),len(pred),'T:',type(targets),len(pred))
        loss = criterion(pred, targets)
        loss_sum += loss.item()
    loss_sum = loss_sum / len(dataloader)
    print('Validation in Epoch [{}], Step [{}], Perplexity: {:5.4f}, Avg Loss: {:.4f}\n'
          .format(epoch+1, i+1, np.exp(loss_sum), loss_sum))

    return -loss_sum

def generate_sample(output_path, epoch, i, model , features, captions, lengths, outputs, BS, vocab):
    # print('Sample generating...')
    # print('caption shape', captions.size()) #(B,seq_len)
    # print('features shape', features.size()) #(B, H*3)
    sampled_ids = model._sample(captions[0][0], features[0])
    sampled_ids = sampled_ids.cpu().data.numpy()

    sampled_ids = sampled_ids[0]
    # print('\nsampled_ids', sampled_ids.shape)
    print(sampled_ids)
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        # if word == '<end>':
        #     break
    Inference = ' '.join(sampled_caption)
    # print(i)
    # if i+1 == 20:
    #     raise NameError('\nEnd')

    outputs, _ = pad_packed_sequence(PackedSequence(outputs, BS), batch_first=True)
    values, indices = outputs[0, :, :].max(1)
    indices = indices.cpu().data.numpy()
    gen_sen = []
    for word_id in indices[:int(lengths[0])]:
        word = vocab.idx2word[word_id]
        gen_sen.append(word)
        if word == '<end>':
            break
    Training = ' '.join(gen_sen)
    # Print out image and generated caption.

    caption = captions[0].cpu().data.numpy()
    ori_sen = []
    for word_id in caption:
        word = vocab.idx2word[word_id]
        ori_sen.append(word)
        if word == '<end>':
            break
    Origin = ' '.join(ori_sen)

    # Print out image and generated caption.
    FILE = output_path + '/sample-EPOCH[{}]-STEP[{}].txt'.format(epoch, i)
    mode = 'a' if os.path.exists(FILE) else 'w'
    with open(FILE, mode) as outf:
        outf.write('\nEPOCH: [{}]/ STEP: [{}]'.format(epoch, i))
        outf.write('\nInference: \n{}'.format(Inference))
        outf.write('\nTraining: \n{}'.format(Training))
        outf.write('\nOrigin: \n{}'.format(Origin))
        outf.write('\nEND EPOCH: [{}]/ STEP: [{}]\n'.format(epoch, i))
    print('Sample generated!!!\n')

if __name__ == '__main__':
    args = parse()
    # Create model and output directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if torch.cuda.is_available():
        with torch.cuda.device(args.gpu):
            torch.manual_seed(1000)
            if args.sample < 0:
                print('Current GPU:', torch.cuda.current_device(),'\n')
                train(args)
            else:
                print('write sample.py')
                raise NotImplementError
    else:
        print('No GPU')