import tensorflow as tf
import numpy as np
from caser import Caser
from interactions import Interactions
import argparse
def get_config():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/ml1m/train.csv')
    parser.add_argument('--val_root', type=str, default='datasets/ml1m/val.csv')
    parser.add_argument('--test_root', type=str, default='datasets/ml1m/test.csv')
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--T', type=int, default=1)  # next_T
    parser.add_argument('--top_k', type=int, default=20)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=1)  # same as T
    parser.add_argument('--decay_rate', default=0.99, type=float)
    # model arguments
    parser.add_argument('--d', type=int, default=50)  # embedding dims
    parser.add_argument('--nv', type=int, default=4)  # the number of filters for vertical convolution
    parser.add_argument('--nh', type=int, default=16)  # the number of filters for horizontal convolution
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--check_dir', type=str, default='save/')
    config = parser.parse_args()
    return config

def test(args):
    data = Interactions(args.test_root)
    data.to_sequence(args.L, args.T)
    sequences_np = data.sequences.sequences
    targets_np = data.sequences.targets
    users_np = data.sequences.user_ids.reshape(-1, 1)
    n_test = sequences_np.shape[0]
    print('total test instances: %d' % n_test)
    num_users = data.num_users
    num_items = data.num_items
    NDCG, HR, MRR = 0.0, 0.0, 0.0
    item_ids = np.zeros((args.batch_size,num_items))
    for i in range(args.batch_size):
        item_ids[i] = np.arange(num_items)
    test_batches = n_test // args.batch_size

    model=Caser(num_users,num_items,args)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.check_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore model from {} successfully!'.format(args.check_dir))
        else:
            print('Restore model from {} failed!'.format(args.check_dir))
            return
        for i in range(test_batches):
            sequences = sequences_np[i * args.batch_size: (i + 1) * args.batch_size]
            targets = targets_np[i * args.batch_size: (i + 1) * args.batch_size]
            users = users_np[i * args.batch_size: (i + 1) * args.batch_size]
            _, top_k_index = model.predict(sess, sequences, users, item_ids)
            hr, ndcg, mrr = 0.0, 0.0, 0.0
            for i in range(args.batch_size):
                cur_user = top_k_index[i]
                for j in range(args.top_k):
                    if targets[i][0] == cur_user[j]:
                        hr += 1
                        mrr += 1 / (1 + j)
                        dcg = 1 / np.log2(1 + 1 + j)
                        idcg = 1 / np.log2(1 + 1)
                        ndcg += dcg / idcg
                        break
            HR += hr / args.batch_size
            NDCG += ndcg / args.batch_size
            MRR += mrr / args.batch_size
    return HR / test_batches, NDCG / test_batches, MRR / test_batches

if __name__=="__main__":
    args=get_config()
    HR,NDCG,MRR=test(args)
    print("test:HR@20-{:.6f}\tNDCG@20-{:.6f}\tMRR@20-{:.6f}".format(HR,NDCG,MRR))