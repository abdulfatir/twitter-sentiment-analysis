import sys
import utils


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
    train = False
    processed_csv = sys.argv[1]
    predictions = classify(processed_csv, test_file=(not train), positive_words=sys.argv[2], negative_words=sys.argv[3])
    if train:
        correct = sum([1 for p in predictions if p[1] == p[2]]) * 100.0 / len(predictions)
        print 'Correct = %.2f%%' % correct
    else:
        utils.save_results_to_csv(predictions, 'baseline.csv')
