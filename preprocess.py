import numpy as np
from string import punctuation
from collections import Counter

vocab_to_int = dict()

def split_text(reviews):

    # removing punctuations from text
    text = ''.join([word for word in reviews if word not in punctuation])
    reviews = text.split('\n')

    text = ' '.join(reviews)
    words = text.split()
    return reviews, words

def create_vocab(words, min_freq):
    
    # creating vocab
    word_counter = Counter()
    for word in words:
        word_counter[word] += 1
    
    vocab = set()
    for word in word_counter.keys():
        if word_counter[word] > min_freq:
            vocab.add(word)          
    return vocab    

def word_to_int(vocab):
    
    # Create your dictionary that maps vocab words to integers here
    index = 1
    global vocab_to_int
    int_to_vocab = dict()

    for word in vocab:
        vocab_to_int[word] = index
        int_to_vocab[index] = word
        index += 1   
    return vocab_to_int

def convert_reviews_to_int(vocab, vocab_to_int, reviews):

    # Convert the reviews to integers, same shape as reviews list, but with integers
    reviews_int = list()

    for review in reviews:
        ints = list()
        for word in review.split():
            if word in vocab:
                ints.append(vocab_to_int[word])
        reviews_int.append(ints)     
    return reviews_int, len(vocab)+1
    
def text_preprocess(text, min_freq=10):
    
    reviews, words = split_text(text)
    vocab = create_vocab(words, min_freq)
    vocab_to_int = word_to_int(vocab)
    reviews_int, vocab_size = convert_reviews_to_int(vocab, vocab_to_int, reviews)
    return reviews_int, vocab_size

    
def convert_labels(labels):
    
    labels = labels.split('\n')

    for i in range(len(labels)):  
        if labels[i] == 'positive':
            labels[i] = 1
        else:        
            labels[i] = 0
    return labels

def truncate(reviews_int, seq_len=200):
    
    features = list()

    for review in reviews_int:
        length = len(review)  
        if length > seq_len:
            features.append(review[:seq_len])
        elif length < seq_len:
            features.append(([0 for i in range(seq_len-length)] + review))
        else:
            features.append(review)
        
    features = np.array(features)
    return features

def convert_new_review(review, seq_len=200):
    
    global vocab_to_int
    review = review.split()
    
    for i in range(len(review)):
        review[i] = vocab_to_int[review[i]]
        
        features = list()
        length = len(review)  
    if length > seq_len:
        features.append(review[:seq_len])
    elif length < seq_len:
        features.append(([0 for i in range(seq_len-length)] + review))
    else:
        features.append(review)
    return features