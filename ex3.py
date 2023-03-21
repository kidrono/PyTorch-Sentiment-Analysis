import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
# import tqdm
import matplotlib.pyplot as plt
from data_loader import SentimentTreeBank

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    sum = np.zeros((1, embedding_dim))
    for word in sent.text:
        if word in word_to_vec:
            sum += word_to_vec[word]
        else:
            sum += np.zeros((1, embedding_dim))
    avg = sum / len(sent.text)
    return torch.from_numpy(avg).view(embedding_dim).float()


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    arr = np.zeros(size, dtype=np.float32)
    arr[ind] = 1
    return arr


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return: average one hot vector
    """
    size = len(word_to_ind)
    sum = np.zeros(size)
    for word in sent.text:
        sum += get_one_hot(size, word_to_ind[word])
    avg = sum / len(sent.text)
    return avg



def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    words_list = list(set(words_list))
    word_dict = {w: num for num, w in enumerate(words_list)}
    return word_dict


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    out = np.zeros((seq_len, embedding_dim))
    for i in range(min(seq_len, len(sent.text))):
        out[i, :] = word_to_vec.get(sent.text[i], np.zeros(embedding_dim))
    return out


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, vec):
        # lstm_out = self.lstm(vec.float().unbind(0))
        # hidden_concat = torch.cat(lstm_out[1][0].unbind(0), 1)

        lstm_out = self.lstm(vec.float())
        hidden_concat = torch.cat(lstm_out[1][0].unbind(0), 1)
        dropped = self.dropout1(hidden_concat)
        return self.fc(dropped)

    def predict(self, vec):
        lstm_out = self.lstm(vec.float())
        hidden_concat = torch.cat(lstm_out[1][0].unbind(0), 1)
        lin_out = self.fc(hidden_concat)
        sig = nn.Sigmoid()(lin_out)
        sig = sig.round()
        return sig


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=1)
        self.linear1

    def forward(self, x):
        return self.linear1(x.float())

    def predict(self, x):
        f = self.forward(x)
        sig = nn.Sigmoid()(f)
        sig = sig.round()
        # return data_loader.get_sentiment_class_from_val_array(sig)
        return sig



# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    accuracy = 1 - ((preds.squeeze() - y.squeeze()).abs().sum() / y.squeeze().size()[0])
    return accuracy


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    loss_sum = 0
    accuracy_sum = 0
    iter_num = 0
    for batch, y in data_iterator:
        y = torch.unsqueeze(y.float(), 1)
        batch = batch.float()
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        accurate = binary_accuracy(model.predict(batch), y)
        accuracy_sum += accurate
        loss_sum += loss
        iter_num += 1
    return accuracy_sum/iter_num, loss_sum/iter_num


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    loss_sum = 0
    accuracy_sum = 0
    data_num = 0
    iter_num = 0
    for batch, y in data_iterator:
        y = torch.unsqueeze(y.float(), 1)
        pred = model(batch)
        loss = criterion(pred, y)
        accurate = binary_accuracy(model.predict(batch), y)
        accuracy_sum += accurate
        loss_sum += loss
        data_num += len(batch)
        iter_num += 1
    return accuracy_sum / iter_num, loss_sum / iter_num


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    preds = torch.empty(1,1)
    for batch, y in data_iter:
        pred = model.predict(batch)
        preds = torch.cat((preds, pred), 0)
    return preds


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.001):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    accuracies = []
    losses = []
    accuracies_validation = []
    losses_validation = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    validation_iterator = data_manager.get_torch_iterator(VAL)

    print("start training")
    for i in range(n_epochs):
        data_iterator = data_manager.get_torch_iterator()
        accuracy, loss = train_epoch(model, data_iterator, optimizer, criterion)
        accuracies.append(accuracy)
        losses.append(loss)
        print(f"iter: {i}, loss: {loss}, accuracy: {accuracy}")

        validation_accuracy, validation_loss = evaluate(model, validation_iterator, criterion)
        accuracies_validation.append(validation_accuracy)
        losses_validation.append(validation_loss)

    dataset = data_manager.sentiment_dataset
    all_test_labels = data_manager.get_labels(data_subset=TEST)
    all_test_preds = get_predictions_for_data(model, data_manager.get_torch_iterator(data_subset=TEST))
    negated_sentences_idx = data_loader.get_negated_polarity_examples(dataset.get_test_set())
    rare_sentences = data_loader.get_rare_words_examples(data_manager.sentences[TEST], dataset)
    idxs_dict = {"rare": rare_sentences, "negated": negated_sentences_idx}
    for data_type, idxs in idxs_dict.items():
        preds = all_test_preds[idxs]
        labels = torch.from_numpy(all_test_labels[idxs])
        y = labels.view(labels.shape[0], 1)
        accuracy = binary_accuracy(preds.squeeze(), y.squeeze())
        loss = criterion(all_test_preds[idxs].squeeze(), y.squeeze())
        print(f"{data_type} loss: {loss}\n{data_type} accuracy: {accuracy}")

    return accuracies, losses, accuracies_validation, losses_validation


def make_graphs(accuracies, losses, accuracies_validation, losses_validation, model_name):
    plt.figure()
    accuracies = [a.detach() for a in accuracies]
    accuracies_validation = [a.detach() for a in accuracies_validation]
    plt.plot(range(len(accuracies)), accuracies, 'r--', label="train")
    plt.plot(range(len(accuracies_validation)), accuracies_validation, 'b-', label="validation")
    plt.title(f"accurecies of {model_name} model")
    plt.legend()
    plt.show(block=False)

    losses = [a.detach() for a in losses]
    losses_validation = [a.detach() for a in losses_validation]
    plt.figure()
    plt.plot(losses, 'r--', label="train")
    plt.plot(losses_validation, 'b-', label="test")
    plt.title(f"losses of {model_name} model")
    plt.legend()
    plt.show(block=False)



def train_log_linear_with_one_hot(n_epochs=20, batch_size=64, lr=1e-3, weight_decay=0.001):
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=batch_size)
    embedding_size = data_manager.get_input_shape()[0]
    model = LogLinear(embedding_size)
    accuracies, losses, accuracies_validation, losses_validation = train_model(model, data_manager, n_epochs, lr, weight_decay=weight_decay)
    make_graphs(accuracies, losses, accuracies_validation, losses_validation, "log linear")






def train_log_linear_with_w2v(n_epochs=20, batch_size=64, lr=1e-3, weight_decay=0.001):
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    print("training log linear with w2v")
    data_manager = DataManager(data_type=W2V_AVERAGE, embedding_dim=300)
    embedding_size = data_manager.get_input_shape()[0]
    model = LogLinear(embedding_size)
    accuracies, losses, accuracies_validation, losses_validation = train_model(model, data_manager, n_epochs, lr, weight_decay=0.001)

    make_graphs(accuracies, losses, accuracies_validation, losses_validation, "log linear W2V")


def train_lstm_with_w2v(n_epochs=4):
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    print("training lstm with w2v")
    hidden_dim = 100
    dropout = 0.5
    n_layers = 1

    lr = 1e-3
    data_manager = DataManager(data_type=W2V_SEQUENCE, embedding_dim=300)
    embedding_size = data_manager.get_input_shape()[-1]
    model = LSTM(embedding_size, hidden_dim, n_layers, dropout)
    accuracies, losses, accuracies_validation, losses_validation = train_model(model, data_manager, n_epochs, lr, weight_decay=0.)

    make_graphs(accuracies, losses, accuracies_validation, losses_validation, "LSTM model")




if __name__ == '__main__':
    print("starting!")
    print("make sure in right pytorch version")
    print(torch.__version__)
    print("starting train_log_linear_with_one_hot")
    # train_log_linear_with_one_hot(n_epochs=20, weight_decay=0.001)
    print("starting train_log_linear_with_w2v")
    train_log_linear_with_w2v(n_epochs=20)
    print("starting train_lstm_with_w2v")
    train_lstm_with_w2v(n_epochs=4)

    print("done!")

    plt.show(block=True)