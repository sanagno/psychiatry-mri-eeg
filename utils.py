from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import numpy as np
import warnings


# total recall
def precision_recall(true, predictions):
    indices = np.where(true == 1)
    total = len(indices[0])
    found = 0
    for i, j in zip(indices[0], indices[1]):
        found += predictions[i][j]

    # total precision, total recall
    return found / np.sum(predictions), found / total


def count_correct(out1, out2, num_classes):
    # out2 is the prediction
    total_tokens = 0
    correct_tokens = 0
    for o1, o2 in zip(out1.astype(np.int32), out2.astype(np.int32)):
        for i, token in enumerate(o1):
            if o1[i] >= num_classes:
                break

            total_tokens += 1
            if o1[i] == o2[i]:
                correct_tokens += 1

    return correct_tokens / total_tokens


def f1_per_class(true, predictions):
    if not issubclass(predictions.dtype.type, np.integer):
        predictions = before_softmax_to_predictions(predictions)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1_scores = list()
        for i in range(true.shape[1]):
#            f1_scores.append(f1_score(true[:, i], predictions[:, i], average='macro'))
            f1_scores.append(f1_score(true[:, i], predictions[:, i]))

    return f1_scores


def multi_label_accuracy(true, predictions):
    if not issubclass(predictions.dtype.type, np.integer):
        predictions = before_softmax_to_predictions(predictions)

    return 1 - np.sum((true - predictions) ** 2) / (true.shape[0] * true.shape[1])


def get_batches(iterable, batch_size=64, do_shuffle=True):
    if do_shuffle:
        iterable = shuffle(iterable)

    length = len(iterable)
    for ndx in range(0, length, batch_size):
        iterable_batch = iterable[ndx: min(ndx + batch_size, length)]
        yield iterable_batch


def get_predictions_from_sequences(sequences, num_classes):
    predictions = np.zeros((sequences.shape[0], num_classes), dtype=np.int32)
    for i, seq in enumerate(sequences.astype(int)):
        for s in seq:
            if s >= num_classes:
                # assume eod_token or something worse
                break
            predictions[i][s] = 1
    return predictions


def before_softmax_to_predictions(predictions):
    return (predictions >= 0).astype(np.int16)


def get_reconstruction_loss(true, predictions, mask):
    loss = np.mean(((true - predictions) ** 2) * mask, axis=1)
    return np.mean(loss, axis=0)
