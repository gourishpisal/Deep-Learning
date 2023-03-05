import argparse, os, time, sys
import numpy as np
import pandas as pd
import random
import string
import pickle
from math import log, log1p

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

from tqdm import tqdm
from nltk.corpus import stopwords

from pytorch_model_summary import summary

stop_words = set(stopwords.words('english'))

# print("Stop Words: ", stop_words)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
# device = 'cpu'

MAX_WORDS_SENTENCE=25
GRADIENT_CLIPPING_PARAM = 1
EPOCHS=50

# Model
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(4096, 256, batch_first = True)
        self.embedding = nn.Embedding(2428, 256)
        self.decoder = nn.LSTM(256*2, 256, batch_first = True)
        self.net = nn.Linear(256, 2428)

    def forward(self, X, y, train=True):
        output_list = []
        # Encoder
        encoder_output, hidden = self.encoder(X)
        idxs = torch.ones(X.shape[0], 1, dtype=torch.long).to(device)
        context = torch.zeros(X.shape[0], 1, 256).to(device)
        # Iterate over max length pre-defined for sentence
        for i in range(MAX_WORDS_SENTENCE):
            # Embed
            input_emb = torch.cat((self.embedding(idxs),context),2)
            # Decoder
            decoder_output, hidden = self.decoder(input_emb, hidden)
            
            # Attention
            attn_energy = torch.tanh(torch.bmm(encoder_output, hidden[0].transpose(0,1).transpose(1,2)))
            attn_weights = F.softmax(attn_energy, dim=1).squeeze(2).unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_output)
            
            # Output index for word
            output = self.net(decoder_output)
            output_list.append(output)
            
            # Teacher Forcing
            teacher_force = random.random()
            if train and teacher_force <= 0.5:
                idxs = y[:,i].unsqueeze(1)
            else:
                _, idxs = torch.max(output, 2)
        return torch.cat(tuple(output_list), 1)

def train_model(model, X, data, optimizer, criterion):
    model.to(device)
    model.train()
    total_loss = 0
    global flag
    for i, (X_pos, y) in enumerate(data):
        X_pos, y = X_pos.to(device), y.to(device)
        X_batch = X[X_pos.tolist(),:,:].to(device)
        
        optimizer.zero_grad()
#         print(summary(model, X_batch.float(), y, print_summary=True,show_hierarchical=True,show_input=True))
#         print(summary(model, X_batch.float(), y, print_summary=True,show_hierarchical=True, show_input=False))
#         sys.exit(0)
        output = model(X_batch.float(), y, True)
        
        loss = criterion(output.view(-1, output.shape[-1]), y.view(-1))
        total_loss += loss

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING_PARAM)

        loss.backward()
        optimizer.step()

    avg_loss = total_loss/len(data.dataset)
    
    return avg_loss.item()

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                print(row[j])
                print(score)
                print(seq + [j])
                candidate = [seq + [j], score - log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

def beam_search_decoder2(post, k):
    """Beam Search Decoder

    Parameters:

        post(Tensor) – the posterior of network.
        k(int) – beam size of decoder.

    Outputs:

        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """

    batch_size, seq_length = post.shape
    log_post = post.log()
    log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
        log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--data', required=True, help="Data Directory - HW2/training_data/")
    parser.add_argument('-o','--output', help='Output File Name - Ex. sample_output.txt')
    parser.add_argument('-l','--labels',required=False, help="Labels File")
    parser.add_argument('-m','--mode', required=True, help='Mode: train/test')
    parser.add_argument('-vmin','--vocabmincount', default=3, help='Minimum Vocab count in the corpus')
    parser.add_argument('-sw','--stopwords', default="no", help="Remove Words: no/yes")
    # parser.add_argument('-maxlen','--maxlength', default=25, help='Maximum length of the sentence')

    args = parser.parse_args()

    if args.mode == "train":
        
#         print(summary(Seq2Seq(), torch.zeros((64, 80, 4096)).to(torch.int64), torch.zeros((64, 25)).to(torch.int64), show_input=True))
#         sys.exit(0)
        
        train_labels = pd.read_json(args.labels)
        # max_len = args.maxlength
        
        ## Shape of Training Labels = (1450, 2)
        # print(train_labels.shape)

        ## Word Counter - Dictionary
        word_count_dict={}
        for idx in range(train_labels.shape[0]):
            processed_captions = []
            for caption in train_labels['caption'][idx]:
                # print(caption)
                words = [word.lower().strip(string.punctuation) for word in caption.split(' ')]
                if args.stopwords=="yes":
                    words = [word for word in words if word not in stop_words]
                # print(words)
                for word in words:
                    if word in word_count_dict: word_count_dict[word]+=1
                    else:   word_count_dict[word]=1
                # print(word_count_dict)
                if len(words)>MAX_WORDS_SENTENCE-2:
                    words = words[:MAX_WORDS_SENTENCE-2]
                processed_captions.append(' '.join(words))
            
            train_labels['caption'][idx] = processed_captions

        ## Retrieve Words with Minimum Vocab Count
        vocabulary_list = ['<pad>', '<bos>', '<eos>', '<unk>']
        vocabulary_list.extend([word for word,count in word_count_dict.items() if count>args.vocabmincount]) 

        print("Vocaublary Length: {}".format(len(vocabulary_list)))

        ## Save vaocabulary as pickle file
        with open('vocab.pkl', 'wb') as file:
            pickle.dump(vocabulary_list,file)

        ## Encoder ::: Load Train Data - Video Features
        train_feat_list = []
        train_feat_dir = os.path.join(args.data, 'feat/')
        for vid in train_labels['id']:
            train_feat_list.append(np.load(train_feat_dir+vid+".npy"))
        
        ## Shape of each video features = (80, 4096)
        # print(train_feat_list[0].shape) 
        
        # with open(os.path.join(args.data,'id.txt'),'r') as file:
        #     data_vids = file.read().strip().split('\n')

        ## Decoder ::: Convert Vocabulary tokens to IDS
        input_vids = []
        targets = []
        for idx in range(train_labels.shape[0]):
            for caption in train_labels['caption'][idx]:
                input_vids.append(idx)
                ## <bos> word1 word2 ... wordN <eos>
                interim_string = [3 if word not in vocabulary_list else vocabulary_list.index(word) for word in caption.split()]
                padding_string = interim_string+[0]*(MAX_WORDS_SENTENCE-len(interim_string)-1)+[2]
                # print(interim_string)
                # print(padding_string)
                targets.append(np.array(padding_string))
            #     break
            # break

        # print(len(targets))
        # print(len(input_vids))

        targets = np.array(targets)
        # # targets = np.vstack(targets).astype(np.float)
        input_vids = np.array(input_vids)
        # print(targets.shape)
        # print(input_vids.shape)
        # print(targets.dtype)
        # targets = torch.from_numpy(targets)
        targets = torch.LongTensor(targets)
        # print(targets.shape)
        
        decoder_train_loader = Data.DataLoader(Data.TensorDataset(torch.tensor(input_vids), targets), 64, shuffle=True)

        ## Encoder ::: Train Features - Tensor Array
        encoder_feat = torch.tensor(np.array(train_feat_list))

#         for i, (X_pos, y) in enumerate(decoder_train_loader):
#             print(i, X_pos, y)
#             X_batch = encoder_feat[X_pos.tolist(),:,:]
#             print(X_batch.shape)
#             print(y.shape)
#             print(X_batch[0].shape)
#             print(X_batch[0])
#             print(X_batch[0][0].shape)
#             print(X_batch[0][0])
#             break
            
#         sys.exit(0)
        
        ## Initiate Sequence to Sequence Model
        model = Seq2Seq().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train
        loss_list = []
        for i in range(1,EPOCHS+1):
            start = time.time()
            loss = train_model(model, encoder_feat, decoder_train_loader, optimizer, criterion)
            print('Epoch:{}  \tLoss:{:.8f}\t\tTime:{:.4f}s'.format(i, loss, time.time()-start))
            loss_list.append(loss)
        torch.save(model.cpu().state_dict(), 'v2t_s2s_model.pt')

    else:
        output_file = args.output
        
        test_feat_dir = os.path.join(args.data,'feat/')
        test_feat_files = os.listdir(test_feat_dir)
        
        ## Load Vocabulary
        with open('vocab.pkl','rb') as file:
            words = pickle.load(file)
            
        ## Test Features
        feat_list = []
        for file in test_feat_files:
            feat_list.append(np.load(test_feat_dir+file))
            
        ## Load Sequence to Sequence Model
        model = Seq2Seq().to(device)
        model.load_state_dict(torch.load('v2t_s2s_model.pt'))
        model.eval()
        
        video_prediction_list = []
        
        # Get predictions
        for i in range(len(feat_list)):
            feat = torch.tensor(feat_list[i]).unsqueeze(0).to(device)
            output = model(feat.float(), None, False).squeeze(0)
#             print(output)
#             print(output.shape)
#             print(output.log)
#             result = beam_search_decoder2(output, 3)
#             print(result)
            _, idx_list = torch.max(output,1)
#             print(idx_list.tolist())
#             print([words[i] for i in idx_list.tolist() if i != 0 and i != 2])
            video_prediction_list.append([test_feat_files[i][:-4], ' '.join([words[i] for i in idx_list.tolist() if i != 0 and i != 2])])
#             break

        # Write results to file
        with open(args.output,'w') as out:
            for i in video_prediction_list:
                out.write('{},{}\n'.format(i[0], i[1]))
