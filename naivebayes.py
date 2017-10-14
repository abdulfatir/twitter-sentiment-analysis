import sys
import pickle
import utils
import random
import numpy as np
from sklearn.naive_bayes import MultinomialNB


def top_n_words(pkl_file_name, N):
    with open(pkl_file_name, 'rb') as pkl_file:
        freq_dist = pickle.load(pkl_file)
    most_common = freq_dist.most_common(N)
    words = {p[0]: i for i, p in enumerate(most_common)}
    return words


def get_feature_vector(tweet):
    feature_vector = []
    words = tweet.split()
    for word in words:
        if vocab.get(word):
            feature_vector.append(word)
    return feature_vector


def extract_features(tweets, batch_size=500, test_file=True, feat_type='presence'):
    num_batches = int(np.ceil(len(tweets) / float(batch_size)))
    for i in xrange(num_batches):
        batch = tweets[i * batch_size: (i + 1) * batch_size]
        features = np.zeros((batch_size, vocab_size))
        labels = np.zeros(batch_size)
        for j, tweet in enumerate(batch):
            if test_file:
                tweet_words = set(tweet[1])
            else:
                tweet_words = set(tweet[2])
                labels[j] = tweet[1]
            for word in tweet_words:
                idx = vocab.get(word)
                if idx:
                    if feat_type == 'presence':
                        features[j, idx] = 1
                    elif feat_type == 'frequency':
                        features[j, idx] += 1
        yield features, labels


def process_tweets(csv_file, test_file=True):
    tweets = []
    print 'Generating feature vectors'
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
            utils.write_status(i + 1, total)
    print '\n'
    return tweets


def split_data(tweets, validation_split=0.1):
    index = int((1 - validation_split) * len(tweets))
    random.shuffle(tweets)
    return tweets[:index], tweets[index:]


if __name__ == '__main__':
    train = True
    np.random.seed(1337)
    train_csv_file = sys.argv[1]
    test_csv_file = sys.argv[2]
    pkl_file_name = sys.argv[3]
    global vocab
    global vocab_size
    vocab_size = 10000
    batch_size = 500
    feat_type = 'presence'
    vocab = top_n_words(pkl_file_name, vocab_size)
    tweets = process_tweets(train_csv_file, test_file=False)
    if train:
        train_tweets, val_tweets = split_data(tweets)
    else:
        random.shuffle(tweets)
        train_tweets = tweets
    del tweets
    print 'Extracting features & training batches'
    clf = MultinomialNB()
    i = 1
    n_train_batches = int(np.ceil(len(train_tweets) / float(batch_size)))
    for training_set_X, training_set_y in extract_features(train_tweets, test_file=False, feat_type=feat_type):
        utils.write_status(i, n_train_batches)
        i += 1
        clf.partial_fit(training_set_X, training_set_y, classes=[0, 1])
    print '\n'
    print 'Testing'
    if train:
        correct, total = 0, len(val_tweets)
        i = 1
        n_val_batches = int(np.ceil(len(val_tweets) / float(batch_size)))
        for val_set_X, val_set_y in extract_features(val_tweets, test_file=False, feat_type=feat_type):
            prediction = clf.predict(val_set_X)
            correct += np.sum(prediction == val_set_y)
            utils.write_status(i, n_val_batches)
            i += 1
        print '\nCorrect: %d/%d = %.4f %%' % (correct, total, correct * 100. / total)
    else:
        del train_tweets
        test_tweets = process_tweets(test_csv_file, test_file=True)
        n_test_batches = int(np.ceil(len(test_tweets) / float(batch_size)))
        predictions = np.array([])
        print 'Predicting batches'
        i = 1
        for test_set_X, _ in extract_features(test_tweets, test_file=True, feat_type=feat_type):
            prediction = clf.predict(test_set_X)
            predictions = np.concatenate((predictions, prediction))
            utils.write_status(i, n_test_batches)
            i += 1
        predictions = [(str(j), int(predictions[j])) for j in range(len(test_tweets))]
        utils.save_results_to_csv(predictions, 'naivebayes.csv')
        print '\nSaved to naivebayes.csv'
