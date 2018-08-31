import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def grasping_cnn(unscaled_images, **conv_kwargs):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc_img1', nh=512, init_scale=np.sqrt(2)))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x, **conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X, processed_x = observation_input(ob_space, nbatch)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class CombiPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X_img, X_robot, X_task, processed_x_img, processed_x_robot, processed_x_task = observation_input(ob_space, nbatch)
            activ = tf.tanh
            processed_x_robot = tf.layers.flatten(processed_x_robot)
            ### policy network
            with tf.variable_scope("pi", reuse=reuse):
                # CNN stream
                pi_h_img = grasping_cnn(processed_x_img, **conv_kwargs)
                # robot_state stream
                pi_h1_robot = activ(fc(processed_x_robot, 'fc1', nh=64, init_scale=np.sqrt(2)))
                pi_h2_robot = activ(fc(pi_h1_robot, 'fc2', nh=64, init_scale=np.sqrt(2)))
                # output stream
                pi_h1 = tf.concat([pi_h_img, pi_h2_robot],1)
                pi_h2 = activ(fc(pi_h1, 'fc3', nh=128, init_scale=np.sqrt(2)))
                self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


            ### value function network
            with tf.variable_scope("vf", reuse=reuse):
                # CNN stream
                vf_h_img = grasping_cnn(processed_x_img, **conv_kwargs)
                # robot_state stream
                vf_h1_robot = activ(fc(processed_x_robot, 'fc1', nh=64, init_scale=np.sqrt(2)))
                vf_h2_robot = activ(fc(vf_h1_robot, 'fc2', nh=64, init_scale=np.sqrt(2)))
                # output stream
                vf_h1 = tf.concat([vf_h_img, vf_h2_robot],1)
                vf_h2 = activ(fc(vf_h1, 'fc3', nh=128, init_scale=np.sqrt(2)))
                vf = fc(vf_h2, 'value', 1)[:, 0]

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X_img:ob['img'], X_robot:ob['robot_state']})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X_img:ob['img'], X_robot:ob['robot_state']})

        self.X_img = X_img
        self.X_robot = X_robot
        self.vf = vf
        self.step = step
        self.value = value

        self.placeholder_dict = {
            "img": X_img,
            "robot_state": X_robot
        }

class AsyncCombiPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X_img, X_robot, X_task, processed_x_img, processed_x_robot, processed_x_task = observation_input(ob_space, nbatch)
            activ = tf.tanh
            processed_x_robot = tf.layers.flatten(processed_x_robot)
            processed_x_task = tf.layers.flatten(processed_x_task)
            ### policy network
            with tf.variable_scope("pi", reuse=reuse):
                # CNN stream
                pi_h_img = grasping_cnn(processed_x_img, **conv_kwargs)
                # robot_state stream
                pi_h1_robot = activ(fc(processed_x_robot, 'fc1', nh=64, init_scale=np.sqrt(2)))
                pi_h2_robot = activ(fc(pi_h1_robot, 'fc2', nh=64, init_scale=np.sqrt(2)))
                # output stream
                pi_h1 = tf.concat([pi_h_img, pi_h2_robot],1)
                pi_h2 = activ(fc(pi_h1, 'fc3', nh=128, init_scale=np.sqrt(2)))
                self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


            ### value function network
            with tf.variable_scope("vf", reuse=reuse):
                # concat robot and task vector
                vf_input = tf.concat([processed_x_robot,processed_x_task], 1)
                vf_h1 = activ(fc(vf_input, 'fc1', nh=64, init_scale=np.sqrt(2)))
                vf_h2 = activ(fc(vf_h1, 'fc2', nh=64, init_scale=np.sqrt(2)))
                # output
                vf = fc(vf_h2, 'value', 1)[:, 0]

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X_img:ob['img'], X_robot:ob['robot_state'],X_task:ob['task_state']})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X_task:ob['task_state'], X_robot:ob['robot_state']})

        self.X_img = X_img
        self.X_robot = X_robot
        self.X_task = X_task
        self.vf = vf
        self.step = step
        self.value = value

        self.placeholder_dict = {
            "img": X_img,
            "robot_state": X_robot,
            "task_state": X_task
        }