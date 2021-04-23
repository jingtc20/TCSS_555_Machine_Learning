#/usr/bin/python3
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import os
from string import punctuation 
import argparse
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from collections import defaultdict
from nltk.tag import pos_tag
from collections import Counter

import math as m
import torch
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# nltk.download()
file_names = []
DATA_DIR = ['positive_polarity', 'negative_polarity']
sub_dirs = []
for sub_dir in DATA_DIR:
    sub_dirs = sub_dirs + [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)  if f.startswith('.') is False]
tmp = []
for sub_dir in sub_dirs:
    tmp = tmp + [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)  if f.startswith('.') is False]
sub_dirs = tmp
for sub_dir in sub_dirs:
    file_names = file_names + [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)  if f.startswith('.') is False]
# print(len(file_names))


class Email(object):
    def __init__(self, data_path=None):
        self.type = 0
        self.words = 0
        self.sentences = []
        self.lemmatizer = Lemmatization()
        if data_path is not None:
            self.load_from_file(data_path)

    def remove_stopwords(self, email):
        sentence = email.split(' ')
        stop_words = set(stopwords.words('english'))
        new_email = []
        for word in sentence:
            for char in word:
                if ord(char) < 97 or ord(char) > 122:
                    word = word.replace(char, '')
            if word not in stop_words and len(word) > 0: 
                new_email.append(word)
        return new_email

    def load_from_file(self, data_path):
        data_path_name = data_path.split('/')
        data_path_name_type = data_path_name[1][0]
        # 1 represents spam email
        self.type = 1.0 if data_path_name_type == 'd' else 0.0
        with open(data_path) as f:
            for line in f.readlines():
                email = line.lower()         
                email = self.remove_stopwords(email)
                for punc in punctuation:
                    if punc in email:
                        email = email.replace(punc, '')           
                new_email = self.lemmatizer.get_lemmatized_email(email)
                self.sentences = new_email
                self.words = len(self.sentences)
        return self.sentences

class Lemmatization(object):
    def __init__(self):
        pass

    def get_pos_tag(self, nltk_tag):  
        if nltk_tag.startswith('J') or nltk_tag.startswith('A'):
            return wordnet.ADJ
        elif nltk_tag.startswith('S'):
            return wordnet.ADJ_SAT
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None
    
    def get_lemmatized_email(self, email):
        lemma = WordNetLemmatizer()
        # Split each line into a list of tokens
        sentence = email
        # Creates a list of tuples. First element is the word, second is the pos_tag
        pos_tags = nltk.pos_tag(sentence)
        # Iterate through each tuple in the list.
        new_email = []
        for tag in pos_tags: 
            word = tag[0]
            pos_tag = self.get_pos_tag(tag[1]) or wordnet.NOUN
            new_email.append(lemma.lemmatize(word, pos_tag))
        return new_email

class Dataset():
    def __init__(self, file_names = None, model = None, input_size = 100):
        self.file_names = file_names
        self.model = model
        self.input_size = input_size
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = file_names[idx]
        email = Email(file_name)
        sentences = email.sentences
        # vectorize the sentences, and convert the vector form to tensor 
        sentences_vector = []
        for j in range(len(sentences)):
            sentences_vector.append(self.model.wv[sentences[j]])
        sentences_tensor = torch.tensor(sentences_vector)
        # make up each review to the longest length in order to do mini batch gradient descent
        residue_tensor = torch.zeros(376 - email.words, self.input_size)
        data_tensor = torch.cat((sentences_tensor, residue_tensor), 0)
        label = torch.tensor(email.type)
        num_words = torch.tensor(email.words)
        return  data_tensor, label, num_words

import torch.nn as nn
class SpamChecker(nn.Module):
    def __init__(self, rnn_input_size, rnn_hidden_size, rnn_num_layers, dropout=0):
        super(SpamChecker, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_input_size = rnn_input_size
        self.rnn_num_layers = rnn_num_layers
        # self.rnn = nn.RNN(rnn_input_size, rnn_hidden_size, self.rnn_num_layers)
        self.rnn = nn.LSTM(rnn_input_size, rnn_hidden_size, self.rnn_num_layers, dropout=dropout, bidirectional=True)
        self.fc_models = nn.Linear(rnn_hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, num_words):
        # rnn_output.shape (376, batch_size, rnn_hidden_size)
        seq_len, batch_size, feature_size = input.shape
        h0 = torch.zeros(self.rnn_num_layers * 2, batch_size, self.rnn_hidden_size).to(device)
        c0 = torch.zeros(self.rnn_num_layers * 2, batch_size, self.rnn_hidden_size).to(device)
        rnn_output, hn = self.rnn(input, (h0, c0))
        outputs = torch.zeros(batch_size, self.rnn_hidden_size * 2)
        for j, num_word in enumerate(num_words):
            outputs[j] = rnn_output[num_word - 1][j]
        outputs = outputs.to(device)
        fc_output = self.fc_models(outputs)
        sigmoid_outputs = self.sigmoid(fc_output)
        final_outputs = torch.reshape(sigmoid_outputs, (-1,))
        return final_outputs

def prob_to_label(p):
    for i in range(len(p)):
        p[i] = 1.0 if p[i] > 0.5 else 0.0
    return p

def get_word2vec_model(total_sentences):
    model = Word2Vec(total_sentences, size=100, min_count=1, workers=4, sg = 1, seed = 1, window = 5)
    model.save("word_2_vec.model")

def main():
    total_sentences = [[]]
    for i in range(len(file_names)):
        total_sentences[0].extend(Email(file_names[i]).sentences)
    input_size = 100
    get_word2vec_model(total_sentences)
    word2vec_model = Word2Vec.load("word_2_vec.model")
    # model = Word2Vec(total_sentences, size=input_size, min_count=1, workers=4, sg = 1)
    dataset = Dataset(file_names[0: 1600], word2vec_model, input_size)
    # split the dataset into 80% trainset and 20% testset
    trainset, testset= train_test_split(dataset, test_size=0.20)
    batch_size = 128
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(testset, batch_size=5,  shuffle=False, num_workers=4)
    
    num_layers = 1
    n_hidden =  128
    dropout = 0.0
    rnn = SpamChecker(input_size, n_hidden, num_layers, dropout)
    rnn = rnn.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    all_losses = []
    train_accuracy = []
    test_accuracy = []
    num_epoch = 150
    num_train_times = 2
    
    start = time.time()
    for epoch in tqdm(range(num_epoch)):
    # for epoch in range(num_epoch):
        loss_sum = 0.0
        
        rnn.train()
        for t in range(num_train_times):
            for i, (input_review, labels, num_words) in enumerate(train_loader):
                input_review = input_review.to(device)
                labels = labels.to(device)
                num_words = num_words.to(device)
                rnn.zero_grad()
                input_review = input_review.permute(1, 0, 2)
                num_words = num_words.tolist()
                outputs = rnn(input_review, num_words)           
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_sum += loss.item()
            
        
        avg_loss = loss_sum / len(train_loader) / float(num_train_times)
        all_losses.append(avg_loss)
        print("epoch: ", epoch, "average loss:", avg_loss)
        lr_scheduler.step()

        correct_num = 0
        total_num = 0
        rnn.eval()
        for i, (input_review, labels, num_words) in enumerate(train_loader):
            outputs_all = []
            input_review = input_review.permute(1, 0, 2).to(device)
            labels = labels.to(device)
            num_words = num_words.tolist()
            outputs = rnn(input_review, num_words)   
            true_labels = np.array(labels.tolist())
            guess_lables = np.array(prob_to_label(outputs).tolist())
            correct_num += np.sum(true_labels==guess_lables)
            total_num += len(guess_lables)
        accuracy_rate = correct_num / total_num
        train_accuracy.append(accuracy_rate)
        print('accuracy rate for training set:',accuracy_rate)

        correct_num = 0
        total_num = 0
        # rnn.eval()
        for i, (input_review, labels, num_words) in enumerate(test_loader):
            outputs_all = []
            input_review = input_review.permute(1, 0, 2).to(device)
            labels = labels.to(device)
            num_words = num_words.tolist()
            outputs = rnn(input_review, num_words)  
            true_labels = np.array(labels.tolist())
            guess_lables = np.array(prob_to_label(outputs).tolist())
            correct_num += np.sum(true_labels==guess_lables)
            total_num += len(guess_lables)
        accuracy_rate = correct_num / total_num
        test_accuracy.append(accuracy_rate)
        print('accuracy rate for test set:',accuracy_rate)

    end = time.time()
    train_time = start - end
    print('training time is:', train_time)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker 
    plt.figure(1)
    plt.plot(list(range(epoch + 1)), all_losses)
    plt.ylabel('loss', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.title('performance graph on loss')
    plt.savefig('graph_mini_GPU.png')

    plt.figure(2)
    plt.plot(list(range(epoch + 1)), train_accuracy)
    plt.ylabel('accuracy of train dataset', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.title('performance graph on accuracy of train data')
    plt.savefig('graph_mini_GPU_train.png')


    plt.figure(3)
    plt.plot(list(range(epoch + 1)), test_accuracy)
    plt.ylabel('accuracy of test dataset', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.title('performance graph on accuracy of test data')
    plt.savefig('graph_mini_GPU__test.png')







if __name__ == '__main__':
    main()







