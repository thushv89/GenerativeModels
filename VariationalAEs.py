import tensorflow as tf
import logging
import numpy as np
import os
import sys
import struct

__author__ = 'Thushan Ganegedara'

'''================================================================
Variational Autoencoders

Work the following way

X - input data
X~ - generated data
f(theta) - Decoder
f(phi) - Encoder
Q - PDF of Encoder
P - PDF of Decoder
z - latent variables

X -> f(theta) -> mu,sigma
z = mu + sigma^(1/2)*epsilon where epsilon ~ N(0,I)
z -> f(phi) -> X~

Objective Function (Maximizing P(X))
log(P(X)) - KL[Q(z|X)||P(z|X)] = E(z~Q)[logP(X|z)] - KL[Q(z|X)||P(z)]
where logP(X|z) = ||X - f(z;theta)||2
and KL[Q(z|X)||P(z)] = -(1/2)*(1+log(sigma^2) - mu^2 - sigma^2) used in practice
Maximize RHS
================================================================'''

logging_level = logging.DEBUG
logging_format = '[%(funcName)s] %(message)s'

in_dim = 784
z_dim = 128

batch_size = 128
learning_rate = 0.001

total_epochs = 5

Q_layers = ['fulcon_1','fulcon_2','fulcon_3','fulcon_4','fulcon_out_mu','fulcon_out_sigma'] # encoder
P_layers = ['fulcon_1','fulcon_2','fulcon_3','fulcon_out'] # decoder

Q_fc1_hyp = {'in':in_dim,'out':1024}
Q_fc2_hyp = {'in':Q_fc1_hyp['out'],'out':768}
Q_fc3_hyp = {'in':Q_fc2_hyp['out'],'out':512}
Q_fc4_hyp = {'in':Q_fc3_hyp['out'],'out':256}
Q_fc_out_mu_hyp = {'in':Q_fc4_hyp['out'],'out':z_dim}
Q_fc_out_sig_hyp = {'in':Q_fc4_hyp['out'],'out':z_dim}

Q_hyperparameters = {
    'fulcon_1':Q_fc1_hyp,'fulcon_2':Q_fc2_hyp,
    'fulcon_3':Q_fc3_hyp,'fulcon_4':Q_fc4_hyp,
    'fulcon_out_mu':Q_fc_out_mu_hyp,'fulcon_out_sigma':Q_fc_out_sig_hyp
}

P_fc1_hyp = {'in':z_dim,'out':256}
P_fc2_hyp = {'in':P_fc1_hyp['out'],'out':512}
P_fc3_hyp = {'in':P_fc2_hyp['out'],'out':1024}
P_fc_out_hyp = {'in':P_fc3_hyp['out'],'out':in_dim}

P_hyperparameters =  {
    'fulcon_1':P_fc1_hyp,'fulcon_2':P_fc2_hyp,
    'fulcon_3':P_fc3_hyp,'fulcon_out':P_fc_out_hyp
}

P_Weights, P_Biases = {},{}
Q_Weights, Q_Biases = {},{}

logger = None

def initialize_encoder_and_decoder():

    logger.info('Initializing the Encoder(Q) and Decoder(P)\n')
    # initializing the decoder
    for pl in P_layers:
        P_Weights[pl] = tf.Variable(tf.truncated_normal(
            [P_hyperparameters[pl]['in'], P_hyperparameters[pl]['out']],
            stddev=2. / (P_hyperparameters[pl]['in'] + P_hyperparameters[pl]['out']))
        ,name = 'w_'+pl)
        P_Biases[pl] = tf.Variable(tf.constant(
            0.0, shape=[P_hyperparameters[pl]['out']]
        ), name='b_'+pl)

    # initializing the encoder
    for ql in Q_layers:
        Q_Weights[ql] = tf.Variable(tf.truncated_normal(
            [Q_hyperparameters[ql]['in'], Q_hyperparameters[ql]['out']],
            stddev=2. / (Q_hyperparameters[ql]['in'] + Q_hyperparameters[ql]['out']))
            , name='w_' + ql)
        Q_Biases[ql] = tf.Variable(tf.constant(
            0.0, shape=[Q_hyperparameters[ql]['out']]
        ), name='b_' + ql)


# encoding function
def Q(x):
    outputs = [x]

    # we calculate the output up to the last hidden layer
    for ql in Q_layers[:-2]:
        logger.debug('Calculating Encoder output for %s',ql)
        outputs.append(tf.nn.relu(tf.matmul(outputs[-1],Q_Weights[ql])+Q_Biases[ql]))
        logger.debug('\tOutput size: %s\n',outputs[-1].get_shape().as_list())
    # calculate mu and sigma
    logger.debug('Calculating Encoder mu')
    outputs_mu = tf.nn.relu(tf.matmul(outputs[-1],Q_Weights['fulcon_out_mu'])+Q_Biases['fulcon_out_mu'])
    logger.debug('\tOutput size: %s\n', outputs_mu.get_shape().as_list())
    logger.debug('Calculating Encoder sigma')
    outputs_sigma=tf.nn.relu(tf.matmul(outputs[-1], Q_Weights['fulcon_out_sigma']) + Q_Biases['fulcon_out_sigma'])
    logger.debug('\tOutput size: %s\n', outputs_sigma.get_shape().as_list())

    outputs.append(outputs_mu)
    outputs.append(outputs_sigma)

    return outputs


# decoding function
def P(z):
    outputs = [z]

    for pl in P_layers[:-1]:
        logger.debug('Calculating Decoder output for %s', pl)
        outputs.append(tf.nn.relu(tf.matmul(outputs[-1], P_Weights[pl]) + P_Biases[pl]))
        logger.debug('\tOutput size: %s\n', outputs[-1].get_shape().as_list())

    outputs.append(tf.nn.sigmoid(tf.matmul(outputs[-1], P_Weights['fulcon_out']) + P_Biases['fulcon_out']))
    return outputs


def loss(x,x_tilde,mu,sigma):

    log_pX_given_z_loss = tf.reduce_sum((x-x_tilde)**2,1)
    kl_div_loss = -tf.reduce_sum(0.5*(1+ tf.log(1e-10+sigma**2) - mu**2 - sigma**2),1)

    l = tf.reduce_mean(log_pX_given_z_loss + kl_div_loss,name='cost')

    return l


def optimize(loss):

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)
    return optimizer

def load_mnist(fname_img):
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows*cols)

    return img

if __name__ == '__main__':

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    graph = tf.Graph()

    dataset = load_mnist('data'+os.sep + 'train-images.idx3-ubyte')
    dataset_size = dataset.shape[0]

    logger.info('Data loaded of size: %s',str(dataset.shape))

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as session:

        tf_x = tf.placeholder(tf.float32,shape=(None,in_dim),name='X')
        tf_epsilon = tf.placeholder(tf.float32,shape=(None,z_dim),name='z')

        initialize_encoder_and_decoder()

        tf_Q_out = Q(tf_x)
        tf_Q_mu,tf_Q_sig = tf_Q_out[-1],tf_Q_out[-2]
        tf_z = tf_Q_mu + tf_Q_sig * tf_epsilon

        tf_P_out = P(tf_z)
        tf_x_tilde = tf_P_out[-1]

        tf_loss = loss(tf_x,tf_x_tilde,tf_Q_mu,tf_Q_sig)
        tf_optimize = optimize(tf_loss)

        session.run(tf.global_variables_initializer())

        for epoch in range(total_epochs):

            for batch_id in range((dataset_size//batch_size) -1):
                batch_data = dataset[batch_id*batch_size:(batch_id+1)*batch_size,:]
                epsilon = np.random.normal(size=(batch_size,z_dim))

                q,p,loss,_,generated_x = session.run([tf_Q_out,tf_P_out,tf_loss,tf_optimize,tf_x_tilde],feed_dict = {tf_x:batch_data,tf_epsilon:epsilon})

                logger.info('='*60)
                logger.info('Epoch: %d',epoch)
                logger.info('Loss: %.5f',loss)
                logger.info('Mean mu: %.2f',np.mean(q[-2]))
                logger.info('Mean sig: %.2f', np.mean(q[-1]))
