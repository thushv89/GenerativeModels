import tensorflow as tf
import logging
import numpy as np
import os
import sys
import struct
from scipy.misc import imsave

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
z_dim = 20

batch_size = 128
learning_rate = 0.0001

total_epochs = 80
persist_count = 6

beta = 0

Q_layers = ['fulcon_1','fulcon_2','fulcon_3','fulcon_out_mu','fulcon_out_sigma'] # encoder
P_layers = ['fulcon_1','fulcon_2','fulcon_3','fulcon_out'] # decoder

# Hyperparameters of the decoder
Q_fc1_hyp = {'in':in_dim,'out':1024}
Q_fc2_hyp = {'in':Q_fc1_hyp['out'],'out':768}
Q_fc3_hyp = {'in':Q_fc2_hyp['out'],'out':512}
Q_fc_out_mu_hyp = {'in':Q_fc3_hyp['out'],'out':z_dim}
Q_fc_out_sig_hyp = {'in':Q_fc3_hyp['out'],'out':z_dim}

Q_hyperparameters = {
    'fulcon_1':Q_fc1_hyp,'fulcon_2':Q_fc2_hyp,
    'fulcon_3':Q_fc3_hyp,
    'fulcon_out_mu':Q_fc_out_mu_hyp,'fulcon_out_sigma':Q_fc_out_sig_hyp
}

# Hyperparameters of the encoder
P_fc1_hyp = {'in':z_dim,'out':512}
P_fc2_hyp = {'in':P_fc1_hyp['out'],'out':768}
P_fc3_hyp = {'in':P_fc2_hyp['out'],'out':1024}
P_fc_out_hyp = {'in':P_fc3_hyp['out'],'out':in_dim}

P_hyperparameters =  {
    'fulcon_1':P_fc1_hyp, 'fulcon_2':P_fc2_hyp,
    'fulcon_3':P_fc3_hyp,
    'fulcon_out':P_fc_out_hyp
}

P_Weights, P_Biases = {},{} # weights and biases of the encoder
Q_Weights, Q_Biases = {},{} # weights and biases of the decoder

logger = None

def initialize_encoder_and_decoder():

    logger.info('Initializing the Encoder(Q) and Decoder(P)\n')
    # initializing the decoder
    for pl in P_layers:
        P_Weights[pl] = tf.Variable(tf.truncated_normal(
            [P_hyperparameters[pl]['in'], P_hyperparameters[pl]['out']],
            stddev=(2. / (P_hyperparameters[pl]['in']))**0.5)
        ,name = 'Pw_'+pl)
        P_Biases[pl] = tf.Variable(tf.constant(
            0.0, shape=[P_hyperparameters[pl]['out']]
        ), name='Pb_'+pl)

    # initializing the encoder
    for ql in Q_layers:
        Q_Weights[ql] = tf.Variable(tf.truncated_normal(
            [Q_hyperparameters[ql]['in'], Q_hyperparameters[ql]['out']],
            stddev=(2. / Q_hyperparameters[ql]['in'])**0.5)
            , name='Qw_' + ql)
        Q_Biases[ql] = tf.Variable(tf.constant(
            0.0, shape=[Q_hyperparameters[ql]['out']]
        ), name='Qb_' + ql)


# encoding function
def Q(x):
    outputs = [x]

    # we calculate the output up to the last hidden layer
    for ql in Q_layers[:-2]:
        logger.debug('Calculating Encoder output for %s',ql)
        outputs.append(tf.nn.elu(tf.matmul(outputs[-1],Q_Weights[ql])+Q_Biases[ql]))
        logger.debug('\tOutput size: %s\n',outputs[-1].get_shape().as_list())

    # calculate mu and sigma
    logger.debug('Calculating Encoder mu')
    outputs_mu = tf.matmul(outputs[-1],Q_Weights['fulcon_out_mu'])+Q_Biases['fulcon_out_mu']
    logger.debug('\tOutput size: %s\n', outputs_mu.get_shape().as_list())

    logger.debug('Calculating Encoder sigma')
    outputs_sigma= tf.matmul(outputs[-1], Q_Weights['fulcon_out_sigma']) + Q_Biases['fulcon_out_sigma']
    logger.debug('\tOutput size: %s\n', outputs_sigma.get_shape().as_list())

    outputs.append(outputs_mu) # [-2] index
    outputs.append(outputs_sigma) # [-1] index

    return outputs_mu,outputs_sigma


# decoding function
def P(z):
    outputs = [z]

    for pl in P_layers[:-1]:
        logger.debug('Calculating Decoder output for %s', pl)
        outputs.append(tf.nn.elu(tf.matmul(outputs[-1], P_Weights[pl]) + P_Biases[pl]))
        logger.debug('\tOutput size: %s\n', outputs[-1].get_shape().as_list())

    outputs.append(tf.matmul(outputs[-1], P_Weights['fulcon_out']) + P_Biases['fulcon_out'])

    return outputs[-1]


def loss(x,x_tilde,mu,logsig):

    # reconstruction loss (Uncomment the one you need and comment the one you don't)
    log_pX_given_z_loss = tf.reduce_sum(tf.square(x-x_tilde),1)
    # binary cross-entropy loss (Uncomment the one you need and comment the one you don't)
    #log_pX_given_z_loss = -tf.reduce_sum(x*tf.log(1e-10 + x_tilde) + (1.0-x)*tf.log(1e-10 + 1.0 - x_tilde), 1)

    kl_div_loss = - 0.5 * tf.reduce_sum(1 + 2*logsig - tf.square(mu) - tf.exp(2*logsig),1)

    P_l2_loss = beta * tf.reduce_sum([tf.nn.l2_loss(w) for w in list(P_Weights.values())])
    Q_l2_loss = beta * tf.reduce_sum([tf.nn.l2_loss(w) for w in list(Q_Weights.values())])
    l = tf.reduce_mean(log_pX_given_z_loss + kl_div_loss,name='cost')
    l = l + P_l2_loss + Q_l2_loss
    return l


def optimize(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer


def sample_gaussian(mu,log_sigma):
    return mu + tf.exp(log_sigma) * tf.random_normal([batch_size,z_dim])


def load_mnist(fname_img):
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows*cols)
        img = img.astype('float32')
        img = img/255.0
        logger.info('After normalize: min max: %.2f %.2f',np.min(img),np.max(img))
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

    logger.info('='*60)
    logger.info('Data loaded of size: %s',str(dataset.shape))
    logger.info('Data min max: %.2f,%.2f',np.min(dataset),np.max(dataset))
    logger.info('='*60)
    assert np.max(dataset)<=1.0 and np.min(dataset)>= 0.0 # make sure data is normalized to 0 - 1

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as session:

        tf_x = tf.placeholder(tf.float32,shape=(None,in_dim),name='X')

        initialize_encoder_and_decoder()

        tf_Q_mu, tf_Q_logsig = Q(tf_x) # output mu(X) and log(sigma(X))

        tf_z = sample_gaussian(tf_Q_mu,tf_Q_logsig) # sample z with reparameterization trick

        tf_x_tilde = P(tf_z) # generated image

        tf_loss = loss(tf_x, tf_x_tilde, tf_Q_mu, tf_Q_logsig)
        tf_optimize = optimize(tf_loss)

        session.run(tf.global_variables_initializer())

        local_dir = 'images'
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        print([v.name for v in tf.trainable_variables()])
        for epoch in range(total_epochs):

            mean_loss = []
            for batch_id in range((dataset_size//batch_size) -1):
                batch_data = dataset[batch_id*batch_size:(batch_id+1)*batch_size,:]
                z_mu, z_logsig, batch_l, _ = session.run([tf_Q_mu,tf_Q_logsig,tf_loss,tf_optimize],
                                                                    feed_dict = {tf_x:batch_data})

                mean_loss.append(batch_l)
                if np.isnan(batch_l):
                    break

            # print out the epoch summary
            if epoch > 0 and epoch%1==0:
                logger.info('='*60)
                logger.info('Epoch: %d (%d batches)',epoch,batch_id)
                logger.info('Loss: %.5f',np.mean(batch_l))
                logger.info('Mean mu: %.8f',np.mean(z_mu))
                logger.info('Mean sig (Log): %.8f', np.mean(np.exp(z_logsig)))

            # test
            if epoch>0 and (epoch+1)%5==0:
                tf.set_random_seed(np.random.randint(0,13254643))
                generated_x = session.run(tf_x_tilde, feed_dict = {tf_z:tf.random_normal([batch_size,z_dim])})

                for save_id in range(persist_count):
                    img_id = np.random.randint(0,generated_x.shape[0])
                    imsave(local_dir+os.sep+'test_img_'+str(epoch+1)+'_'+str(save_id)+'.png',generated_x[img_id,:].reshape(28,28))