import utils

# Classifies a tweet based on the number of positive and negative words in it

TRAIN_PROCESSED_FILE = 'train-processed.csv'
TEST_PROCESSED_FILE = 'test-processed.csv'
POSITIVE_WORDS_FILE = '../dataset/positive-words.txt'
NEGATIVE_WORDS_FILE = '../dataset/negative-words.txt'
TRAIN = True


def classify(processed_csv, test_file=True, **params):
    positive_words = utils.file_to_wordset(params.pop('positive_words'))
    negative_words = utils.file_to_wordset(params.pop('negative_words'))
    predictions = []
    with open(processed_csv, 'r') as csv:
        for line in csv:
            if test_file:
                tweet_id, tweet = line.strip().split(',')
            else:
                tweet_id, label, tweet = line.strip().split(',')
            pos_count, neg_count = 0, 0
            for word in tweet.split():
                if word in positive_words:
                    pos_count += 1
                elif word in negative_words:
                    neg_count += 1
            # print pos_count, neg_count
            prediction = 1 if pos_count >= neg_count else 0
            if test_file:
                predictions.append((tweet_id, prediction))
            else:
                predictions.append((tweet_id, int(label), prediction))
    return predictions


if __name__ == '__main__':
    if TRAIN:
        predictions = classify(TRAIN_PROCESSED_FILE, test_file=(not TRAIN), positive_words=POSITIVE_WORDS_FILE, negative_words=NEGATIVE_WORDS_FILE)
        correct = sum([1 for p in predictions if p[1] == p[2]]) * 100.0 / len(predictions)
        print 'Correct = %.2f%%' % correct
    else:
        predictions = classify(TEST_PROCESSED_FILE, test_file=(not TRAIN), positive_words=POSITIVE_WORDS_FILE, negative_words=NEGATIVE_WORDS_FILE)
        utils.save_results_to_csv(predictions, 'baseline.csv')
