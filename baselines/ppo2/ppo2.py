import os
import os.path as osp
import time
from collections import deque

import joblib
import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, n_batch_act, n_batch_train,
                 n_steps, ent_coef, vf_coef, max_grad_norm, remember_states):
        """
        :param n_batch_act: size of inference batch (more than 1 because of stacked envs)
        :param n_batch_train: size of train batch
        :param n_steps: length of rollouts for recurrent/stateful policies
        """
        sess = tf.get_default_session()

        act_model = policy(sess=sess,
                           ob_space=ob_space,
                           ac_space=ac_space,
                           n_batch=n_batch_act,
                           n_steps=1,
                           reuse=False)
        train_model = policy(sess=sess,
                             ob_space=ob_space,
                             ac_space=ac_space,
                             n_batch=n_batch_train,
                             n_steps=n_steps,
                             reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLD_NEG_LOG_PAC = tf.placeholder(tf.float32, [None])
        OLD_V_PRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIP_RANGE = tf.placeholder(tf.float32, [])

        neg_log_pac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        v_pred = train_model.vf
        v_pred_clipped = OLD_V_PRED + tf.clip_by_value(train_model.vf - OLD_V_PRED, - CLIP_RANGE, CLIP_RANGE)
        vf_losses1 = tf.square(v_pred - R)
        vf_losses2 = tf.square(v_pred_clipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(OLD_NEG_LOG_PAC - neg_log_pac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approx_kl = .5 * tf.reduce_mean(tf.square(neg_log_pac - OLD_NEG_LOG_PAC))
        clip_frac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIP_RANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        if remember_states:
            memory_loss = .5 * tf.reduce_mean(tf.square(train_model.qf - R))
            loss += memory_loss

        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, clip_range, obs, returns, masks, actions, values, neg_log_pacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            feed_dict = {A: actions,
                         ADV: advs,
                         R: returns,
                         LR: lr,
                         CLIP_RANGE: clip_range,
                         OLD_NEG_LOG_PAC: neg_log_pacs,
                         OLD_V_PRED: values}
            if isinstance(train_model.X, dict):
                for k in train_model.X:
                    feed_dict[train_model.X[k]] = obs[k]
            else:
                feed_dict[train_model.X] = obs
            if states is not None:
                feed_dict[train_model.S] = states
                feed_dict[train_model.M] = masks
            fetch = [pg_loss, vf_loss, entropy, approx_kl, clip_frac, _train]
            if remember_states:
                fetch = [memory_loss] + fetch
            return sess.run(fetch, feed_dict)[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        if remember_states:
            self.loss_names += ['prediction_loss']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neg_log_pacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neg_log_pacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neg_log_pacs.append(neg_log_pacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info: ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neg_log_pacs = np.asarray(mb_neg_log_pacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        last_gae_lam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                next_nonterminal = 1.0 - self.dones
                next_values = last_values
            else:
                next_nonterminal = 1.0 - mb_dones[t + 1]
                next_values = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * next_values * next_nonterminal - mb_values[t]
            mb_advs[t] = last_gae_lam = delta + self.gamma * self.lam * next_nonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neg_log_pacs)),
                mb_states, ep_infos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def const_fn(val):
    def f(_):
        return val

    return f


def learn(*, policy, env, n_steps, total_time_steps, ent_coef, lr,
          remember_states,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, n_mini_batches=4, n_opt_epochs=4, clip_range=0.2,
          save_interval=0, load_path=None):
    """
    :param n_steps: number of steps per rollout (rollouts do not terminate at end of episode)
    :param total_time_steps: total number of time steps across all environments before termination
    :param ent_coef: amount to weight entropy
    :param lr: either a constant learning rate or a function of (1 - % done)
    :param vf_coef: amount to weigh value function loss
    :param max_grad_norm: global grad clip
    :param log_interval: how frequently to perform logging
    :param n_mini_batches: number of slices to chop each batch into
    :param n_opt_epochs: number of training updates to perform per batch
    :param clip_range: ppo clipping
    :param save_interval: how often to save the model
    :param load_path: where to load the model from
    :return: trained model
    """
    if isinstance(lr, float):
        lr = const_fn(lr)
    else:
        assert callable(lr)
    if isinstance(clip_range, float):
        clip_range = const_fn(clip_range)
    else:
        assert callable(clip_range)
    total_time_steps = int(total_time_steps)

    n_env = env.num_envs
    n_batch = n_env * n_steps  # total number of steps in a batch
    n_batch_train = n_batch // n_mini_batches  # total number of steps used in training update

    def make_model():
        return Model(policy=policy, ob_space=env.observation_space,
                     ac_space=env.action_space,
                     n_batch_act=n_env,
                     n_batch_train=n_batch_train,
                     n_steps=n_steps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm,
                     remember_states=remember_states)

    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=model, nsteps=n_steps, gamma=gamma, lam=lam)

    ep_info_buf = deque(maxlen=100)
    global_t_start = time.time()

    n_updates = total_time_steps // n_batch
    for update in range(1, n_updates + 1):
        assert n_batch % n_mini_batches == 0
        n_batch_train = n_batch // n_mini_batches
        t_start = time.time()
        frac = 1.0 - (update - 1.0) / n_updates
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()  # pylint: disable=E0632
        ep_info_buf.extend(epinfos)
        mb_loss_vals = []
        if states is None:  # nonrecurrent version
            indices = np.arange(n_batch)
            for _ in range(n_opt_epochs):
                np.random.shuffle(indices)
                for start in range(0, n_batch, n_batch_train):
                    end = start + n_batch_train
                    mb_indices = indices[start:end]
                    slices = (arr[mb_indices] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mb_loss_vals.append(model.train(lr(frac), clip_range(frac), *slices))
        else:  # recurrent version
            assert n_env % n_mini_batches == 0
            env_indices = np.arange(n_env)
            flat_indices = np.arange(n_env * n_steps).reshape(n_env, n_steps)
            envs_per_batch = n_batch_train // n_steps
            for _ in range(n_opt_epochs):
                np.random.shuffle(env_indices)
                for start in range(0, n_env, envs_per_batch):
                    end = start + envs_per_batch
                    mb_env_indices = env_indices[start:end]
                    mb_flat_indices = flat_indices[mb_env_indices].ravel()
                    slices = (arr[mb_flat_indices] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mb_states = states[mb_env_indices]
                    mb_loss_vals.append(model.train(lr(frac), clip_range(frac), *slices, mb_states))

        lossvals = np.mean(mb_loss_vals, axis=0)
        tnow = time.time()
        fps = int(n_batch / (tnow - t_start))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * n_steps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * n_batch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safe_mean([epinfo['r'] for epinfo in ep_info_buf]))
            logger.logkv('eplenmean', safe_mean([epinfo['l'] for epinfo in ep_info_buf]))
            logger.logkv('time_elapsed', tnow - global_t_start)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
