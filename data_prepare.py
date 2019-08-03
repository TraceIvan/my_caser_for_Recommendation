import pandas as pd
import numpy as np
import os
TRAINFILE='datasets/ml1m/train.csv'
VALIDFILE='datasets/ml1m/val.csv'
TESTFILE='datasets/ml1m/test.csv'
ITEM_MAPFILE='datasets/ml1m/item2id.map'
USER_MAPFILE='datasets/ml1m/user2id.map'
def prepare(file):
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
    num_users = len(user_map)
    num_items = len(item_map)
    data = pd.read_csv(file, sep='\t')
    data['item'] = data['item'].map(item_map)
    data['user'] = data['user'].map(user_map)
    data = data.sort_values(by=['user','Time'])
    print(data.head(5))
    user_ids=list()
    item_ids=list()
    user_ids = data['user'].to_list()
    item_ids = data['item'].to_list()

if __name__=='__main__':
    prepare(TRAINFILE)