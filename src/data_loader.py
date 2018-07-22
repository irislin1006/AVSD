import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import json
import nltk

import operator
from functools import reduce
from build_vocab import Vocabulary

class AVSDDataset(data.Dataset):
    def __init__(self,
                 img_path='/home/iris1006/avsd/data/images/',
                 aud_path='/home/iris1006/avsd/data/audio/i3d_flow',
                 text_path='/home/iris1006/avsd/data/annotations/train_set.json',
                 vocab=None,
                 transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            img_path: image directory.
            aud_path: audio directory.
            json: annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.(no need here)
        """
        self.img_rgb_root = img_path + 'i3d_rgb/'
        self.img_vgg_root = img_path + 'vggish/'
        self.aud_root = aud_path + '/'
        self.annotations = json.load(open(text_path, 'r'))['dialogs']
        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image, audio, dialog(w/o summary), caption)."""
        annotations = self.annotations[index]
        vocab = self.vocab

        image_file = annotations['image_id'] + '.npy'
        rgb_path = os.path.join(self.img_rgb_root, image_file)
        vgg_path = os.path.join(self.img_vgg_root, image_file)
        audio_path = os.path.join(self.aud_root, image_file)
        image_rgb, image_vgg, audio = self.img_aud_process(rgb_path, vgg_path, audio_path)

        # Convert text (string) to word ids.
        summary, dialog, caption = annotations['summary'], annotations['dialog'], annotations['caption']
        summary, dialog, caption = self.test_process(vocab, summary, dialog, caption)

        if len(dialog) != 10:
            txt = '/home/iris1006/avsd/output/weird_data.txt'
            mode = 'a' if os.path.exists(txt) else 'w'
            with open(txt, mode) as outf:
                outf.write('/////////////////////')
                outf.write('name: {}\n'.format(annotations['image_id']))
                outf.write('summary: {}\n'.format(annotations['summary']))
                outf.write('caption: {}\n'.format(annotations['caption']))
                outf.write('dialog: {}\n'.format(annotations['dialog']))
                outf.write('\\\\\\\\\\\\\\\\\\\\\\')
            dialog = dialog[:10]

        return image_rgb, audio, dialog, caption

    def __len__(self):
        return len(self.annotations)

    def img_aud_process(self, rgb_path, vgg_path, aud_path):
        """i.e. rgb (138, "2048"), aud (137, "2048"), vgg (46, "128") """
        image_rgb = np.load(rgb_path)
        image_vgg = np.load(vgg_path)
        audio = np.load(aud_path)
        return torch.from_numpy(image_rgb).float(), torch.from_numpy(image_vgg).float(), torch.from_numpy(audio).float()
        # return image_rgb, image_vgg, audio

    def test_process(self, vocab, summary_text, dialog, caption_text):
        summary, qa_pairs, caption = [], [], []
        for i, qa in enumerate(dialog):
            q = qa['question']
            a = qa['answer']
            qa_pair = q+' '+a
            qa_tokens = qa_pair.strip().split()

            qa_pair = []
            qa_pair.extend([vocab(token) for token in qa_tokens])
            qa_pairs.append(qa_pair)

        summary_tokens = summary_text.strip().split()
        summary.extend([vocab(token) for token in summary_tokens])
        caption_tokens = caption_text.strip().split()
        caption.extend([vocab(token) for token in caption_tokens])
        return summary, qa_pairs, caption

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, audio, dialog(w/o summary), caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, audio, dialog(w/o summary), caption).
            - image: torch tensor of shape (VF, 2048).
            - audio: torch tensor of shape (AF, 2048).
            - dialog: list of 10 qa pairs with variable lengths
            - caption: tuple of 1 sentence with variable lengths
    Returns:
        images: torch tensor of shape (batch_size, VF!=AF, 2048).
        audios: torch tensor of shape (batch_size, AF!=VF, 2048).
        caption: torch tensor of shape (batch_size, cap_padded_length) == target
        dialoges: torch tensor of shape (batch_size, 10, padded_length)
        lengths:
    """
    # Sort a data list by caption length (descending order).
    # bugDATA = data
    data.sort(key=lambda x: len(x[-1]), reverse=True)

    image_rgbs, audios, dialogs, captions = zip(*data)

    # oi, oa, od, oc = image_rgbs, audios, dialogs, captions

    _lengths = {}
    # Images & Audio padding
    rgbs_lengths, auds_lengths = [len(x) for x in image_rgbs], [len(x) for x in audios]
    max_rgbs_fea, max_auds_fea = max(rgbs_lengths), max(auds_lengths)
    rgbs, auds = [], []
    for im, au in zip(image_rgbs, audios):
        _rgbs = torch.cat((im, torch.zeros(max_rgbs_fea - im.size(0), im.size(1))),dim=0)
        rgbs.append(_rgbs)
        _auds = torch.cat((au, torch.zeros(max_auds_fea - au.size(0), au.size(1))),dim=0)
        auds.append(_auds)

    texts = reduce(operator.concat, dialogs)

    lengths = [len(x) + 2 for x in texts]
    cap_lengths = [len(x) + 2 for x in captions]
    max_seq_len = max(lengths)
    cap_max_seq_len = max(cap_lengths)

    texts = [[1] + s + [2] + [0 for _ in range(max_seq_len - len(s) - 2)] for s in texts]
    cap_texts =[[1] + s + [2] + [0 for _ in range(cap_max_seq_len - len(s) - 2)] for s in captions]

    # Tensors
    image_rgbs = torch.stack(rgbs, dim=0).view(-1, max_rgbs_fea, 2048)
    image_rgbs  = image_rgbs.
    audios = torch.stack(auds, dim=0).view(-1, max_auds_fea, 2048)
    captions = torch.LongTensor(cap_texts).view(-1, cap_max_seq_len)
    dialoges = torch.LongTensor(texts).view(-1, 10, max_seq_len)

    _lengths['image_rgbs'] = torch.LongTensor(rgbs_lengths)#.view(-1, 1)
    _lengths['audios'] = torch.LongTensor(auds_lengths)#.view(-1, 1)
    _lengths['dialoges'] = torch.LongTensor(lengths).view(-1, 10)
    _lengths['captions'] = torch.LongTensor(cap_lengths)#.view(-1, 1)

    # if len(image_rgbs) != 32:
    #     oi, oa, od, oc = zip(*bugDATA)
    #
    #     print('i',type(oi), type(oa), type(od), type(oc),'\n')

    return image_rgbs, audios, captions, dialoges, _lengths

def get_loader(img_path, aud_path, text_path, vocab, batch_size, shuffle, num_workers,
               transform=None, fixed_len=False):

    AVSD = AVSDDataset(img_path=img_path, aud_path=aud_path, text_path=text_path,
                       vocab=vocab,transform=transform)
    collate = collate_fn if not fixed_len else collate_fn_fixed
    data_loader = torch.utils.data.DataLoader(dataset=AVSD,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate)
    return data_loader

def prepare_all_loaders(batch_size, vocab, fixed_len=False, is_flat=False):
    IMG_PATH = '/home/iris1006/avsd/data/images/'
    AUD_PATH = '/home/iris1006/avsd/data/audio/i3d_flow'
    TEXT_PATH = '/home/iris1006/avsd/data/annotations/{}_set.json'
    # Load dataloaders
    trainloader = get_loader(IMG_PATH, AUD_PATH, TEXT_PATH.format('train'), vocab,batch_size=batch_size,
                             shuffle=True,num_workers=5, fixed_len=fixed_len)
    valloader = get_loader(IMG_PATH, AUD_PATH, TEXT_PATH.format('valid'), vocab,batch_size=batch_size,
                           shuffle=True,num_workers=5, fixed_len=fixed_len)
    testloader = get_loader(IMG_PATH, AUD_PATH, TEXT_PATH.format('test'), vocab,batch_size=1,
                            shuffle=False,num_workers=5, fixed_len=fixed_len)
    return trainloader, valloader, testloader

if __name__ == '__main__':

    prepare_all_loaders(32, None)