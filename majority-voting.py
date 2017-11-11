import glob
import numpy as np
import utils

# Takes majority vote of a number of CSV prediction files.

NUM_PREDICTION_ROWS = 200000


def main():
    csvs = glob.glob('results/*.csv')
    predictions = np.zeros((NUM_PREDICTION_ROWS, 2))
    for csv in csvs:
        with open(csv, 'r') as f:
            lines = f.readlines()[1:]
            current_preds = np.array([int(l.split(',')[1]) for l in lines])
            predictions[range(NUM_PREDICTION_ROWS), current_preds] += 1
    print predictions[:50]
    predictions = np.argmax(predictions, axis=1)
    results = zip(map(str, range(NUM_PREDICTION_ROWS)), predictions)
    utils.save_results_to_csv(results, 'majority-voting.csv')


if __name__ == '__main__':
    main()
