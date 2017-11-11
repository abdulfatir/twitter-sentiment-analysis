import numpy as np
import sys
from keras.models import load_model, Model
import utils
from keras.preprocessing.sequence import pad_sequences

# Extracts dense vector features from penultimate layer of CNN model.

FREQ_DIST_FILE = '../train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = '../train-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = '../train-processed.csv'
TEST_PROCESSED_FILE = '../test-processed.csv'
GLOVE_FILE = './dataset/glove-seeds.txt'
dim = 200


def get_glove_vectors(vocab):
    print 'Looking for GLOVE seeds'
    glove_vectors = {}
    found = 0
    with open(GLOVE_FILE, 'r') as glove_file:
        for i, line in enumerate(glove_file):
            utils.write_status(i + 1, 0)
            tokens = line.strip().split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    print '\n'
    return glove_vectors


def get_feature_vector(tweet):
    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_tweets(csv_file, test_file=True):
    tweets = []
    labels = []
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
                tweets.append(feature_vector)
            else:
                tweets.append(feature_vector)
                labels.append(int(sentiment))
            utils.write_status(i + 1, total)
    print '\n'
    return tweets, np.array(labels)


if __name__ == '__main__':
    train = len(sys.argv) == 1
    np.random.seed(1337)
    vocab_size = 90000
    batch_size = 500
    max_length = 40
    filters = 600
    kernel_size = 3
    vocab = utils.top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)
    glove_vectors = get_glove_vectors(vocab)
    tweets, labels = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    tweets = pad_sequences(tweets, maxlen=max_length, padding='post')
    shuffled_indices = np.random.permutation(tweets.shape[0])
    tweets = tweets[shuffled_indices]
    labels = labels[shuffled_indices]
    model = load_model(sys.argv[1])
    model = Model(model.layers[0].input, model.layers[-3].output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()
    test_tweets, _ = process_tweets(TEST_PROCESSED_FILE, test_file=True)
    test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
    predictions = model.predict(test_tweets, batch_size=1024, verbose=1)
    np.save('test-feats.npy', predictions)
    predictions = model.predict(tweets, batch_size=1024, verbose=1)
    np.save('train-feats.npy', predictions)
    np.savetxt('train-labels.txt', labels)
