import sys


def file_to_wordset(filename):
    ''' Converts a file with a word per line to a Python set '''
    words = []
    with open(filename, 'r') as f:
        for line in f:
            words.append(line.strip())
    return set(words)


def write_status(i, total):
    ''' Writes status of a process to console '''
    sys.stdout.write('\r')
    sys.stdout.write('Processing %d/%d' % (i, total))
    sys.stdout.flush()


def save_results_to_csv(results, csv_file):
    ''' Save list of type [(tweet_id, positive)] to csv in Kaggle format '''
    with open(csv_file, 'w') as csv:
        csv.write('id,prediction\n')
        for tweet_id, pred in results:
            csv.write(tweet_id)
            csv.write(',')
            csv.write(str(pred))
            csv.write('\n')
