#/usr/bin/python3
from gensim.test.utils import common_texts, get_tmpfile
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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
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
        self.label = 0
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
        self.label = 1.0 if data_path_name_type == 'd' else 0.0
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


class TfidfDataset():
    def __init__(self, tf_idf_array, emails):
        self.tf_idf_array = tf_idf_array
        self.emails = emails
        
    def __len__(self):
        assert(self.tf_idf_array.shape[0] == len(self.emails))
        return self.tf_idf_array.shape[0]
    
    def __getitem__(self, idx):
        return self.tf_idf_array[idx], self.emails[idx].label


class SpamChecker(nn.Module):
    def __init__(self, input_size):
        super(SpamChecker, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(input_size, input_size // 2),
                        nn.ReLU(),
                        nn.Linear(input_size // 2, 1),
                        nn.Sigmoid()
                        )            

    def forward(self, input):
        return self.model(input).squeeze(1)


def prob_to_label(p):
    for i in range(len(p)):
        p[i] = 1.0 if p[i] > 0.5 else 0.0
    return p


def main():
    emails, labels, corpus, sentences  = list(), list(), list(), list()
    vocabulary = set()
    for file_name in file_names:
        emails.append(Email(file_name))
    for email in emails:
        labels.append(email.label)
        vocabulary.update(email.sentences)
        corpus.append(" ".join(email.sentences))
    # calculate the tf_idf for each word in the review
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
        ('tfid', TfidfTransformer())]).fit(corpus)
    tf_array = pipe['count'].transform(corpus).toarray()
    idf_array = pipe['tfid'].idf_
    td_idf = tf_array * idf_array
    print(td_idf)
    print(tf_array)
    print(idf_array)
    exit()
    # create the dataset and split in into 80% trainset and 20% testset
    dataset = TfidfDataset(td_idf, emails)
    trainset, testset= train_test_split(dataset, test_size=0.20)
    batch_size = 256
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(testset, batch_size=5,  shuffle=False, num_workers=4)


    num_vocabulary = tf_array.shape[1]
    fc_model = SpamChecker(num_vocabulary)
    fc_model = fc_model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fc_model.parameters(), lr = 0.01 , momentum=0.9) 
    lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    all_losses, train_accuracy, test_accuracy, train_time = [], [], [], []
    num_epoch = 30 
    
    # train the fc_model
    for epoch in range(num_epoch):
        loss_sum = 0.0
        start_time = time.time()
        fc_model.train()
        for i, (input_review, labels) in enumerate(train_loader):
            fc_model.zero_grad()
            input_review = input_review.to(device).float()
            labels = labels.to(device).float()
            outputs = fc_model(input_review)           
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_sum += loss.item()         
        avg_loss = loss_sum / len(train_loader) 
        all_losses.append(avg_loss)
        print("epoch: ", epoch, "average loss:", avg_loss)
        lr_scheduler.step()
        end_time = time.time()
        train_time.append(end_time - start_time)

        #calculate the accuracy rate for training set
        correct_num = 0
        total_num = 0
        fc_model.eval()
        for i, (input_review, labels) in enumerate(train_loader):
            input_review = input_review.to(device).float()
            labels = labels.to(device).float()
            outputs = fc_model(input_review)   
            true_labels = np.array(labels.tolist())
            guess_lables = np.array(prob_to_label(outputs).tolist())
            correct_num += np.sum(true_labels==guess_lables)
            total_num += len(guess_lables)
        accuracy_rate = correct_num / total_num
        train_accuracy.append(accuracy_rate)
        print('accuracy rate for training set:',accuracy_rate)

        #calculate the accuracy rate for test set
        correct_num = 0
        total_num = 0
        fc_model.eval()
        for i, (input_review, labels) in enumerate(test_loader):
            input_review = input_review.to(device).float()
            labels = labels.to(device).float()
            outputs = fc_model(input_review)  
            true_labels = np.array(labels.tolist())
            guess_lables = np.array(prob_to_label(outputs).tolist())
            correct_num += np.sum(true_labels==guess_lables)
            total_num += len(guess_lables)
        accuracy_rate = correct_num / total_num
        test_accuracy.append(accuracy_rate)
        print('accuracy rate for test set:',accuracy_rate)      


    # plot each graph
    import matplotlib.pyplot as plt
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

    plt.figure(4)
    plt.plot(list(range(epoch + 1)), train_time)
    plt.ylabel('training time(s)', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.title('performance graph on training time')
    plt.savefig('graph_mini_time.png')

if __name__ == '__main__':
    main()







