"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
TRAINFILE='datasets/ml1m/train.csv'
VALIDFILE='datasets/ml1m/val.csv'
TESTFILE='datasets/ml1m/test.csv'
ITEM_MAPFILE='datasets/ml1m/item2id.map'
USER_MAPFILE='datasets/ml1m/user2id.map'

class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.

    Parameters
    ----------

    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """

    def __init__(self, file_path,
                 user_map=None,
                 item_map=None):

        if os.path.exists(ITEM_MAPFILE):
            item_map = {}
            with open(ITEM_MAPFILE, 'r') as fi:
                lines = fi.readlines()
                for line in lines:
                    k, v = line.strip().split('\t')
                    item_map[int(k)] = int(v)
        else:
            train = pd.read_csv(TRAINFILE, sep='\t')
            valid = pd.read_csv(VALIDFILE, sep='\t')
            test = pd.read_csv(TESTFILE, sep='\t')
            data = pd.concat([train, valid, test])
            items = data.item.unique()
            item_map = dict(zip(items, range(items.size)))
            with open(ITEM_MAPFILE, 'w') as fo:
                for k, v in item_map.items():
                    fo.write(str(k) + '\t' + str(v) + '\n')
        if os.path.exists(USER_MAPFILE):
            user_map = {}
            with open(USER_MAPFILE, 'r') as fi:
                lines = fi.readlines()
                for line in lines:
                    k, v = line.strip().split('\t')
                    user_map[int(k)] = int(v)
        else:
            train = pd.read_csv(TRAINFILE, sep='\t')
            valid = pd.read_csv(VALIDFILE, sep='\t')
            test = pd.read_csv(TESTFILE, sep='\t')
            data = pd.concat([train, valid, test])
            users = data.user.unique()
            user_map = dict(zip(users, range(users.size)))
            with open(USER_MAPFILE, 'w') as fo:
                for k, v in user_map.items():
                    fo.write(str(k) + '\t' + str(v) + '\n')
        num_user = len(user_map)
        num_item = len(item_map)
        data = pd.read_csv(file_path, sep='\t')
        data['item'] = data['item'].map(item_map)
        data['user'] = data['user'].map(user_map)
        data = data.sort_values(by=['user', 'Time'])
        #print(data.head(5))
        user_ids = data['user'].to_list()
        item_ids = data['item'].to_list()


        self.num_users = num_user
        self.num_items =num_item

        self.user_ids = np.array(user_ids)
        self.item_ids = np.array(item_ids)

        self.user_map = user_map
        self.item_map = item_map

        self.sequences = None
        self.test_sequences = None

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        # change the item index start from 1 as 0 is used for padding in sequences
        for k, v in self.item_map.items():
            self.item_map[k] = v + 1
        self.item_ids = self.item_ids + 1
        self.num_items += 1

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        for i, (uid,
                item_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           indices,
                                                           max_sequence_length)):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid

        self.sequences = SequenceInteractions(sequence_users, sequences, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences)


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences,
                 targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, window_size, step_size=1):
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), 'constant')


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length):
            yield (user_ids[i], seq)
