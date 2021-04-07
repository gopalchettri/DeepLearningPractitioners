# for Computing rank-1 and rank-5

# import the necessary packages
import numpy as np

def rank5_accuracy(predictions, labels):
    # initializing the rank-1 and rank-5 accuracies
    rank1 = 0
    rank5 = 0

    # loop over the predictions and ground-truth labels
    for (p, gt) in zip(predictions, labels):
        # sort the probabilities by their index in descending
        # order so that the more confident guessses are at the
        # front of the list
        p = np.argsort(p)[::-1]

        # check if the ground-truth label is in the top-5
        # predictions
        if gt in p[:5]:
            rank5 += 1
        
        # check if the ground-truth is the #1 prediction
        if gt == p[0]:
            rank1 += 1
        
        # compute the final rank-1 and rank-5 accuracies
        rank1 /= float(len(labels))
        rank5 /=float(len(labels))

        # retrun a tuple of the rank-1 and rank-5 accuracies
        return (rank1, rank5)