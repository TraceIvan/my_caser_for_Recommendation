import argparse
from time import time

from caser import Caser
from interactions import Interactions
from utils import *


class Recommender(object):
    """
    args: args,Model-related arguments, like latent dimensions.
    """
    def __init__(self, args=None):
        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.args = args

        # learning related
        self._batch_size = self.args.batch_size
        self._n_iter = self.args.n_iter
        self._neg_samples = self.args.neg_samples

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()
        self._top_k=args.top_k
    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.args)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init) 
        

    def fit(self, train, val, verbose=False):
        """
        The general training loop to fit the model
        Parameters
        ----------
        train: :class:`spotlight.interactions.Interactions`
            training instances, also contains test sequences
        val: :class:`spotlight.interactions.Interactions`
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """

        # convert to sequences, targets and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]

        print('total training instances: %d' % n_train)

        if not self._initialized:
            self._initialize(train)

            
        start_epoch = 0
        for epoch_num in range(start_epoch, self._n_iter):

            t1 = time()
            users_np, sequences_np, targets_np = shuffle(users_np,
                                                         sequences_np,
                                                         targets_np)

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            step_loss = 0.0
            best_HR=0.0
            for (minibatch_num,(batch_users,batch_sequences,batch_targets,batch_negatives)) \
                    in enumerate(minibatch(users_np,sequences_np,targets_np,negatives_np,batch_size=self._batch_size)):
                items_to_predict = np.concatenate((batch_targets, batch_negatives), 1)
                loss,global_step = self._net.train(self.sess,
                                             batch_sequences,
                                             batch_users,
                                             items_to_predict)
                step_loss+=loss
                if global_step%1000==0:
                    print('epoch-{}\tstep-{}\tloss-{:.6f}'.format(epoch_num+1,global_step,step_loss/global_step))
                if verbose and global_step  % 10000 == 0:
                    t2 = time()
                    HR, NDCG, MRR = self.predict(val)
                    output_str = "Epoch %d step %d [%.1f s]\tloss=%.6f,HR@20=%.6f, " \
                                 "NDCG@20=%.6f, MRR@20=%.6f,[%.1f s] " % (epoch_num + 1,global_step,
                                                                          t2 - t1, step_loss/global_step,
                                                                          HR, NDCG, MRR, time() - t2)
                    print(output_str)
                    if HR>best_HR:
                        best_HR=HR
                        ckpt_path = self.args.check_dir + 'model.ckpt'
                        self._net.saver.save(self.sess, ckpt_path, global_step=global_step)
                        print("model saved to {}".format(ckpt_path))





                
    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}

        Parameters
        ----------

        users: array of np.int64
            sequence users
        interactions: :class:`spotlight.interactions.Interactions`
            training instances, used for generate candidates
        n: int
            total number of negatives to sample for each sequence
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]

        return negative_samples

    
    def predict(self, val, item_ids=None):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.

        Parameters
        ----------

        user_id: int
           users id for which prediction scores needed.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        """
        sequences_np = val.sequences.sequences
        targets_np = val.sequences.targets
        users_np = val.sequences.user_ids.reshape(-1, 1)
        n_val = sequences_np.shape[0]
        print('total validation instances: %d' % n_val)
        NDCG,HR,MRR=0.0,0.0,0.0
        item_ids = np.zeros((self._batch_size,self._num_items))
        for i in range(self._batch_size):
            item_ids[i]=np.arange(self._num_items)
        valid_batches = n_val // self._batch_size
        for i in range(valid_batches):
            sequences = sequences_np[i * self._batch_size: (i + 1) * self._batch_size]
            targets = targets_np[i * self._batch_size: (i + 1) * self._batch_size]
            users = users_np[i * self._batch_size: (i + 1) * self._batch_size]
            _, top_k_index = self._net.predict(self.sess,sequences,users,item_ids)
            hr, ndcg, mrr =0.0,0.0,0.0
            for i in range(self._batch_size):
                cur_user = top_k_index[i]
                for j in range(self._top_k):
                    if targets[i][0] == cur_user[j]:
                        hr += 1
                        mrr += 1 / (1 + j)
                        dcg = 1 / np.log2(1 + 1 + j)
                        idcg = 1 / np.log2(1 + 1)
                        ndcg += dcg / idcg
                        break
            HR += hr/self._batch_size
            NDCG += ndcg/self._batch_size
            MRR += mrr/self._batch_size
        return HR/valid_batches,NDCG/valid_batches,MRR/valid_batches

    
def main():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/ml1m/train.csv')
    parser.add_argument('--val_root', type=str, default='datasets/ml1m/val.csv')
    parser.add_argument('--test_root', type=str, default='datasets/ml1m/test.csv')
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--T', type=int, default=1)#next_T
    parser.add_argument('--top_k',type=int,default=20)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=1)#same as T
    parser.add_argument('--decay_rate', default=0.99, type=float)
    # model arguments
    parser.add_argument('--d', type=int, default=50)#embedding dims
    parser.add_argument('--nv', type=int, default=4)#the number of filters for vertical convolution
    parser.add_argument('--nh', type=int, default=16)#the number of filters for horizontal convolution
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--check_dir',type=str,default='save/')
    config = parser.parse_args()

    # set seed
    set_seed(config.seed)

    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)

    val = Interactions(config.val_root)
    val.to_sequence(config.L,config.T)

    print(config)
    # fit model
    model = Recommender(args=config)
    model.fit(train, val, verbose=True)

    
if __name__ == '__main__':
    main()