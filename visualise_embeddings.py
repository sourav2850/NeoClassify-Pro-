import numpy as np
import random
import sklearn.decomposition
import sklearn.manifold
import sklearn.cluster
import torch
import torch.nn as nn
import tqdm.notebook as tq
import sys
from transformers import AutoModelForSequenceClassification

import matplotlib.pyplot as plt

import datasets
import models

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

def as_masks(arr):
    '''
    Used for Task 2b Challenge.
    Takes an array of integer class predictions,
    and returns a list of boolean masks over `arr`
    such that:

        masks = as_masks(arr)
        arr[masks[n]]

    will select all instances of predictions for the `n`th class.
    This can then be used to select from a parallel array with
    different information.
    e.g.

        arr = np.array([0, 0, 1, 1, 3, 1, 0])
        masks = as_masks(arr)
        # (0 for False, 1 for True below)
        # masks[0] =   [1, 1, 0, 0, 0, 0, 1]
        # masks[1] =   [0, 0, 1, 1, 0, 1, 0]
        # masks[2] =   [0, 0, 0, 0, 0, 0, 0]
        # masks[3] =   [0, 0, 0, 0, 1, 0, 0]
        embeds = ... # some array shaped [7, 128]
        # embeds[masks[0]] will select all embeds that were for class 0
    '''
    n_classes = arr.max()+1
    one_hot = np.eye(n_classes)[arr]
    return [m == 1 for m in one_hot.T]

def collect_outputs(dl, model, sequenceClassificationModel):
    '''
    Given a dataloader `dl` and a `model`,
    pushes all examples from `dl` through `model`,
    collects the results into a single np array, and uses the labels
    to create masks for each class of the results.
    The sequenceClassificationModel flag is used to determine if we should average
    pool the results. Since if the tokenClassificationModel is used instead
    then we should average the results.
    '''
    collected_outputs = []
    collected_labels = []
    desc = 'Passing data through model'
    with torch.no_grad():
        for texts, labels in tq.tqdm(dl, total=len(dl), desc=desc):
            texts = texts.to(device)
            collected_labels.append(labels)
            out = model(texts)
            if sequenceClassificationModel == False:
              collected_outputs.append(torch.mean(out['logits'][:,1:,], dim=1).cpu().numpy())
            else:
              collected_outputs.append(out['logits'].cpu().numpy())

    collected_labels_np = np.concatenate(collected_labels)
    results_np = np.concatenate(collected_outputs)

    masks = as_masks(collected_labels_np)

    return results_np, masks

def fit_kmeans(embeddings, n_classes=4):
    '''
    Fits kmeans to `embeddings`, producing `n_classes` clusters
    then matches each embedding to it's nearest cluster, and returns
    masks for each cluster of the embeddings.
    '''
    kmeans = sklearn.cluster.KMeans(n_classes)
    class_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    pred = kmeans.fit_predict(embeddings)
    masks = as_masks(pred)
    return class_names, masks

def make_plottable(embeddings):
    '''
    Reduces the dimensionality of embeddings to 2 dimensions so that it can be
    plotted.
    Please be aware that this is a very lossy operation;
    the purpose of TSNE is to reduce the dimensionality of the embedding
    to 2D for visualization. TSNE is the most popular technique for dimensionality
    reduction for visualizing high dimensional data in 2D.
    '''
    # TODO Task 2b - Create an instance TSNE using the following
    #                sklearn library class:
    #   - sklearn.manifold.TSNE
    # Fill in the following to create the tsne objects: 
    #     tsne = sklearn....
    # Then use the above created object to transform the given embeddings 
    # to the reduced dimensions by first fitting the embeddings using by tsne.
    #     plottable = tsne.fit_transform(...)

    return plottable

def plot_classifications(class_names, masks, arr, title):
    '''
    For each point in arr, plot it in a scatter plot,
    colouring it by the class indicated in `masks`.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, mask in enumerate(masks):
        nm = class_names[i]
        ax.scatter(arr[mask][:, 0], arr[mask][:, 1], label=nm, alpha=0.2)
    leg = ax.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.set_title(title)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()

def mk_plots(sentence_len, model_fname=None, sequenceClassificationModel = True):
    # Set random seed to ensure consistent results
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Prepare data
    # TODO Task 2b - Create a dataset and dataloader over the validation set.
    # ds = ..
    # dl = ..



    # TODO Task 2b - Instantiate two kinds of pre-trained 'distilbert-base-uncased' models one
    #                for sequence classification and another for token classification.
    # if sequenceClassificationModel == True:
    #  model = AutoModelFor ...
    # else:
    #  model = AutoModelFor ...
    #
    # Note for Task 2b CHALLENGE where we try to visualize the embeddings generated
    # from a fine-tuned classification model make sure you use the correct kind of model.
    #
    # 

    model.eval()

    if model_fname is not None:
        # TODO Task 2b CHALLENGE -
        #  - Load model weights,
        #  - Use `collect_outputs` to collect the predictions before replacing
        #    the classifier
        #  - Use `as_masks` on the predictions to get masks per class
        #     (store as pred_masks)
    else:
        model.to(device)

    # TODO Task 2b - Our goal is to visualise the pre-classification embeddings.
    #                i.e. the activations that the model gives to the final 
    #                classification layer.                
    #                To extract the pre-classification embeddings, set the 
    #                model's classifier to something that does not perform any 
    #                calculations at all.
    #                So the classifier layer becomes like a pass through layer.
    #                Hint: think about which matrix operation when multiplied 
    #                      another matrix equals itself.  
    #                      E.g.  A x M = A 
    #                       (what type of matrx M has this property?)
    # model.classifier = ....

    print("Collecting embeddings...")
    embeds, label_masks = collect_outputs(dl, model)
    print("Reducing dimensionality of embeddings...")
    plottable = make_plottable(embeds)

    with open('/content/data/txt/classes.txt') as f:
        class_names = [line.rstrip('\n') for line in f]

    plot_classifications(class_names, label_masks, plottable, 'True Labels')
    if model_fname is None:
        print("Fitting kmeans...")
        kmeans_names, kmeans_masks = fit_kmeans(embeds)
        plot_classifications(
            kmeans_names, kmeans_masks, plottable, 'Clustered Labels')
    else:
        # TODO Task 2b CHALLENGE -
        #   Plot the embeddings labelled by the predictions
