import tensorflow as tf
import numpy as np
class Caser(object):
    """
    num_users: int,Number of users.
    num_items: int,Number of items.
    model_args: args,Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, args):
        super(Caser, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.args = args

        # init args
        self.batch_size=self.args.batch_size
        self.L = self.args.L
        self.T = self.args.T        
        self.dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.learning_rate = self.args.learning_rate
        self.l2 = self.args.l2
        self.drop_ratio = self.args.drop
        self.decay_rate=self.args.decay_rate
        self.top_k=self.args.top_k
        # for horizontal conv layer
        self.lengths = [i + 1 for i in range(self.L)]

        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=20)

    def build_model(self):
        """
        """
        self.sequences = tf.placeholder(tf.int32, [None, self.L])#[batch_size,L]
        self.users = tf.placeholder(tf.int32, [None, 1])#[batch_size,1]
        self.items = tf.placeholder(tf.int32, [None, 2*self.T])#[batch_size,2T]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_training = tf.placeholder(tf.bool)
                                             
        # user and item embeddings
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = tf.Variable(initializer([self.num_users, self.dims]))#[num_users,dims]
        self.item_embeddings = tf.Variable(initializer([self.num_items, self.dims]))#[num_items,dims]
        
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = tf.Variable(initializer([self.num_items, self.dims+self.dims]))#[num_users,2dims]
        self.b2 = tf.Variable(initializer([self.num_items, 1]))#[num_items,1]
        
        item_embs = tf.nn.embedding_lookup(self.item_embeddings, self.sequences)#[batch_size,L,dims]
        item_embs = tf.reshape(item_embs, [-1, self.L, self.dims, 1])#[batch_size,L,dims,1]
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.users)#[batch_size,1,dims]
        user_emb = tf.reshape(user_emb, [-1, self.dims])#[batch_size,dims]
        
        # vertical convolution layers
        if self.n_v:
            out_v = tf.layers.conv2d(item_embs, 
                                     self.n_v, 
                                     [self.L, 1], 
                                     activation=tf.nn.relu)#[batch_size,1,dims,n_v]
            out_v = tf.contrib.layers.flatten(out_v)#[batch_size,dims*n_v]
            
        # horizontal convolution layers
        out_hs = list()
        if self.n_h:
            for h in self.lengths:
                conv_out = tf.layers.conv2d(item_embs, 
                                            self.n_h, 
                                            [h, self.dims], 
                                            activation=tf.nn.relu)#[batch_size,L-h+1,1,n_h]
                conv_out = tf.reshape(conv_out, [-1, self.L-h+1, self.n_h])#[batch_size,L-h+1,n_h]
                pool_out = tf.layers.max_pooling1d(conv_out, [self.L-h+1], 1)#[batch_size,1,n_h]
                pool_out = tf.squeeze(pool_out, 1)#[batch_size,n_h]
                out_hs.append(pool_out)
            out_h = tf.concat(out_hs, 1)#[batch_size,L*n_h]
            
        # concat two convolution layers    
        out = tf.concat([out_v, out_h], 1)#[batch_size,dims*n_v+L*n_h]
        
        # fully-connected layer
        z = tf.layers.dense(out, self.dims, activation=tf.nn.relu)#[batch_size,dims]
        z = tf.layers.dropout(z, self.drop_ratio, self.is_training)
        x = tf.concat([z, user_emb], 1)#[batch_size,2*dims]
        x = tf.reshape(x, [-1, 1, 2*self.dims])#[batch_size,1,2*dims]

        w2 = tf.nn.embedding_lookup(self.W2, self.items)#[batch_size,2*T,2*dims]
        b2 = tf.nn.embedding_lookup(self.b2, self.items)#[batch_size,2*T,1]
        b2 = tf.squeeze(b2, 2)#[batch_size,2*T]

        # training with negative samples
        pred = tf.squeeze(tf.matmul(x, tf.transpose(w2, perm=[0,2,1])), 1) + b2#[batch_size,2*T]
        self.target_pred, negative_pred = tf.split(pred, 2, axis=1)#[batch_size,T],[batch_size,T]
    
        # loss
        positive_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.target_pred)))
        negative_loss = -tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(negative_pred)))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2

        self.loss = positive_loss + negative_loss + l2_loss

        # optimizer
        #self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.num_items // self.batch_size,self.decay_rate)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
    
        # For test and val
        self.all_items = tf.placeholder(tf.int32, [None, self.num_items])        
        test_w2 = tf.nn.embedding_lookup(self.W2, self.all_items)
        test_b2 = tf.nn.embedding_lookup(self.b2, self.all_items)        
        test_b2 = tf.reshape(test_b2, [-1, self.num_items])
        self.test_val_pred = tf.reduce_sum(tf.multiply(x, test_w2), axis=2) + test_b2
        prediction_scaled = tf.nn.softmax(self.test_val_pred)
        top_k_value, top_k_index = tf.nn.top_k(prediction_scaled, k=self.top_k)
        self.top_k_index = tf.reshape(top_k_index, [-1, self.top_k])
    
    def train(self, sess, seq_var, user_var, item_var):
        loss, _,global_step  = sess.run([self.loss, self.train_op,self.global_step], feed_dict={self.sequences: seq_var,
                                                                   self.users: user_var,
                                                                   self.items: item_var,
                                                                   self.is_training: True})
        return loss,global_step

    def predict(self, sess, seq_var, user_var, item_var):
        user_var = np.reshape([user_var], [-1, 1])
        item_var = np.reshape(item_var, [-1, self.num_items])
        pred,top_k_index = sess.run([self.test_val_pred,self.top_k_index], feed_dict={self.sequences: seq_var,
                                                       self.users: user_var,
                                                       self.all_items: item_var,
                                                       self.is_training: False})
        return pred,top_k_index