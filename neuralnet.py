from keras.models import Sequential, load_model
from keras.layers import Dense
import sys
import utils
import random
import numpy as np

# Performs classification using an MLP/1-hidden-layer NN.

FREQ_DIST_FILE = '../train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = '../train-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = '../train-processed.csv'
TEST_PROCESSED_FILE = '../test-processed.csv'
TRAIN = True
UNIGRAM_SIZE = 15000
VOCAB_SIZE = UNIGRAM_SIZE
USE_BIGRAMS = False
if USE_BIGRAMS:
    BIGRAM_SIZE = 10000
    VOCAB_SIZE = UNIGRAM_SIZE + BIGRAM_SIZE
FEAT_TYPE = 'frequency'


def get_feature_vector(tweet):
    uni_feature_vector = []
    bi_feature_vector = []
    words = tweet.split()
    for i in xrange(len(words) - 1):
        word = words[i]
        next_word = words[i + 1]
        if unigrams.get(word):
            uni_feature_vector.append(word)
        if USE_BIGRAMS:
            if bigrams.get((word, next_word)):
                bi_feature_vector.append((word, next_word))
    if len(words) >= 1:
        if unigrams.get(words[-1]):
            uni_feature_vector.append(words[-1])
    return uni_feature_vector, bi_feature_vector


def extract_features(tweets, batch_size=500, test_file=True, feat_type='presence'):
    num_batches = int(np.ceil(len(tweets) / float(batch_size)))
    for i in xrange(num_batches):
        batch = tweets[i * batch_size: (i + 1) * batch_size]
        features = np.zeros((batch_size, VOCAB_SIZE))
        labels = np.zeros(batch_size)
        for j, tweet in enumerate(batch):
            if test_file:
                tweet_words = tweet[1][0]
                tweet_bigrams = tweet[1][1]
            else:
                tweet_words = tweet[2][0]
                tweet_bigrams = tweet[2][1]
                labels[j] = tweet[1]
            if feat_type == 'presence':
                tweet_words = set(tweet_words)
                tweet_bigrams = set(tweet_bigrams)
            for word in tweet_words:
                idx = unigrams.get(word)
                if idx:
                    features[j, idx] += 1
            if USE_BIGRAMS:
                for bigram in tweet_bigrams:
                    idx = bigrams.get(bigram)
                    if idx:
                        features[j, UNIGRAM_SIZE + idx] += 1
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


def build_model():
    model = Sequential()
    model.add(Dense(500, input_dim=VOCAB_SIZE, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def evaluate_model(model, val_tweets):
    correct, total = 0, len(val_tweets)
    for val_set_X, val_set_y in extract_features(val_tweets, feat_type=FEAT_TYPE, test_file=False):
        prediction = model.predict_on_batch(val_set_X)
        prediction = np.round(prediction)
        correct += np.sum(prediction == val_set_y[:, None])
    return float(correct) / total


if __name__ == '__main__':
    np.random.seed(1337)
    unigrams = utils.top_n_words(FREQ_DIST_FILE, UNIGRAM_SIZE)
    if USE_BIGRAMS:
        bigrams = utils.top_n_bigrams(BI_FREQ_DIST_FILE, BIGRAM_SIZE)
    tweets = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    if TRAIN:
        train_tweets, val_tweets = utils.split_data(tweets)
    else:
        random.shuffle(tweets)
        train_tweets = tweets
    del tweets
    print 'Extracting features & training batches'
    nb_epochs = 5
    batch_size = 500
    model = build_model()
    n_train_batches = int(np.ceil(len(train_tweets) / float(batch_size)))
    best_val_acc = 0.0
    for j in xrange(nb_epochs):
        i = 1
        for training_set_X, training_set_y in extract_features(train_tweets, feat_type=FEAT_TYPE, batch_size=batch_size, test_file=False):
            o = model.train_on_batch(training_set_X, training_set_y)
            sys.stdout.write('\rIteration %d/%d, loss:%.4f, acc:%.4f' %
                             (i, n_train_batches, o[0], o[1]))
            sys.stdout.flush()
            i += 1
        val_acc = evaluate_model(model, val_tweets)
        print '\nEpoch: %d, val_acc:%.4f' % (j + 1, val_acc)
        random.shuffle(train_tweets)
        if val_acc > best_val_acc:
            print 'Accuracy improved from %.4f to %.4f, saving model' % (best_val_acc, val_acc)
            best_val_acc = val_acc
            model.save('best_model.h5')
    print 'Testing'
    del train_tweets
    del model
    model = load_model('best_model.h5')
    test_tweets = process_tweets(TEST_PROCESSED_FILE, test_file=True)
    n_test_batches = int(np.ceil(len(test_tweets) / float(batch_size)))
    predictions = np.array([])
    print 'Predicting batches'
    i = 1
    for test_set_X, _ in extract_features(test_tweets, feat_type=FEAT_TYPE, batch_size=batch_size, test_file=True):
        prediction = np.round(model.predict_on_batch(test_set_X).flatten())
        predictions = np.concatenate((predictions, prediction))
        utils.write_status(i, n_test_batches)
        i += 1
    predictions = [(str(j), int(predictions[j]))
                   for j in range(len(test_tweets))]
    utils.save_results_to_csv(predictions, '1layerneuralnet.csv')
    print '\nSaved to 1layerneuralnet.csv'
