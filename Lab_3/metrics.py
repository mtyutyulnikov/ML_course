def binary_classification_metrics(prediction, ground_truth):
    TP = ((prediction == 1) & (ground_truth == 1)).sum()
    FP = ((prediction == 1) & (ground_truth == 0)).sum()
    FN = ((prediction == 0) & (ground_truth == 1)).sum()
    TN = ((prediction == 0) & (ground_truth == 0)).sum()

    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    accuracy = (prediction == ground_truth).sum()/len(prediction)
    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    return (prediction == ground_truth).sum()/len(prediction)
