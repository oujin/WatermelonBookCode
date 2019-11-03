import numpy as np
import numpy.random as rdm


def stratified_sampling(data, ratio):
    """
    Parameters:
    ----------
    data: 2-D array-like
        e.g. numpy([[feature1, feature2, ..., featureN, label], ...]).
    ratio: int or float, optional
        the ratio of data trained over all data.

    Returns:
    ----------
    datasets: tuple of 2-D array-like
        e.g. (train_sets, validate_sets).
    """
    train_samples = np.empty((0, data.shape[1]))
    validate_samples = np.empty((0, data.shape[1]))
    labels = list(set(data[:, -1]))
    for label in labels:
        data_with_label = data[data[:, -1] == label]
        n_samples = max(int(len(data_with_label) * ratio), 1)
        indices = np.arange(len(data_with_label))
        train_indices = rdm.choice(indices, n_samples, replace=False)
        validate_indices = np.setdiff1d(indices, train_indices)
        train_samples = np.vstack(
            train_samples, data_with_label[train_indices])
        validate_samples = np.vstack(
            validate_samples, data_with_label[validate_indices])
    return (train_samples, validate_samples)


def hold_out(model, data, ratio=0.8):
    """
    Parameters:
    ----------
    model: object
        parameter model must have the methods: train, predict, etc.
    data: 2-D array-like
        e.g. numpy([[feature1, feature2, ..., featureN, label], ...]).
    ratio: int or float, optional
        the ratio of data trained over all data, default: 0.8.

    Returns:
    ----------
    accuracy: float
        the accuracy of the model ran.
    """
    if not isinstance(ratio, int) and not isinstance(ratio, float):
        raise('Parameter ratio should be in the type of float or integer.')
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise('Gotten no data.')
    train_samples, validate_samples = stratified_sampling(data, ratio)
    model.train(train_samples[:, :-1], train_samples[:, -1])
    pred = model.predict(validate_samples[:, :-1])
    acc = np.sum(
        np.abs(pred == validate_samples[:, -1])) / validate_samples.shape[0]
    return acc


def k_fold_cross_validate(model, data, k=5):
    """
    Parameters:
    ----------
    model: object
        parameter model must have the methods: train, predict, etc.
    data: 2-D array-like
        e.g. numpy([[feature1, feature2, ..., featureN, label], ...]).
    k: int, optional
        here we use k fold cross validation method, default: 5.

    Returns:
    ----------
    accuracy: float
        the accuracy of the model ran.
    """
    # divide into k fold
    datasets = []
    samples, remained = None, data
    while k > 1:
        samples, remained = stratified_sampling(remained, 1 / k)
        datasets.append(samples)
        k -= 1
    datasets.append(remained)
    # train and validate
    accs = []
    for i in range(len(datasets)):
        train_samples = np.vstack(datasets[:i] + datasets[i+1:])
        validate_samples = datasets[i]
        model.train(train_samples[:, :-1], train_samples[:, -1])
        pred = model.predict(validate_samples[:, :-1])
        accs.append(
            np.sum(
                np.abs(pred == validate_samples[:, -1])) /
            validate_samples.shape[0]
        )
    return np.mean(accs)


def bootstrapping(model, data, n_samples):
    """
    Parameters:
    ----------
    model: object
        parameter model must have the methods: train, predict, etc.
    data: 2-D array-like
        e.g. numpy([[feature1, feature2, ..., featureN, label], ...]).
    n_samples: int
        the times to sample from the data with replacement.

    Returns:
    ----------
    accuracy: float
        the accuracy of the model ran.
    """
    indices = np.arange(len(data))
    train_indices = rdm.choice(indices, n_samples, replace=True)
    validate_indices = np.setdiff1d(indices, train_indices)
    if len(validate_indices) <= 0:
        return bootstrapping(model, data, n_samples)
    train_samples = data[train_indices]
    validate_samples = data[validate_indices]
    model.train(train_samples[:, :-1], train_samples[:, -1])
    pred = model.predict(validate_samples[:, :-1])
    acc = np.sum(
        np.abs(pred == validate_samples[:, -1])) / validate_samples.shape[0]
    return acc
