import numpy as np
import tensorflow as tf

global_step = 0  # globalなstep数


class Worker(object):
    def __init__(self, sess, global_server, env, model, thread_id, args):
        """
        各スレッド毎の処理を行うWorker．
        :param sess:            tensorflowのセッション
        :param global_server:   global shared parameter
        :param env:             環境
        :param model:           モデル
        :param thread_id:       スレッドのID
        :param args:            パラメータ群
        """
        self.sess = sess
        self.global_server = global_server
        self.env = env
        self.model = model
        self.thread_id = thread_id

        # パラメータ初期化
        self.tmax = args.tmax
        self.batch_size = args.batch_size
        self.discount_fact = args.discount_fact
        self.entropy_weight = args.entropy_weight

        # 行動数
        self.num_actions = self.env.action_space.n

        # モデルの重み（各スレッドごとに保持するlocalのモデル）
        self.weights = self.model.model.trainable_weights

        # global shared parameterを更新する処理を構築
        self.A, self.R, self.apply_grads = self.build_training_op()

        # global shared parameterと重みを同期する処理を構築
        self.sync_parameter = self.build_sync_op()

    def build_training_op(self):
        """
        global shared parameterを更新する処理を構築
        """
        # 行動と報酬のバッチ
        A = tf.placeholder(tf.int32, [None])
        R = tf.placeholder(tf.float32, [None])

        # 政策のloss
        log_prob = tf.log(tf.reduce_sum(self.model.p_out *
                                        tf.one_hot(A, depth=self.num_actions),
                                        axis=1, keepdims=True))
        p_loss = tf.reduce_mean(-log_prob *
                                tf.stop_gradient(R - self.model.v_out))
        # 政策のentropy
        entropy = tf.reduce_mean(tf.reduce_sum(self.model.p_out *
                                               tf.log(self.model.p_out),
                                               axis=1, keepdims=True))
        # 価値関数のloss
        v_loss = tf.reduce_mean(tf.square(R - self.model.v_out))

        # total loss
        loss = p_loss + entropy * self.entropy_weight + v_loss * 0.5

        # global shared paremeterにgradientを反映する処理
        grads = tf.gradients(loss, self.weights)
        apply_grads = self.global_server.optimizer.apply_gradients(
            zip(grads, self.global_server.weights))

        return A, R, apply_grads

    def build_sync_op(self):
        """
        global shared parameterと重みを同期する処理を構築
        """
        weights = self.model.weights
        src_weights = self.global_server.weights
        sync_parameter = [weight.assign(src_weight) for weight, src_weight
                          in zip(weights, src_weights)]
        return sync_parameter

    def train(self):
        """
        各スレッド毎に学習を実行
        """
        # globalなstep数
        global global_step

        # 環境初期化
        obs = self.env.reset()

        # メインループ
        local_step = 0
        total_reward = 0
        episode_num = 0
        episode_len = 0
        while global_step < self.tmax:
            # global shared parameterと重みを同期
            self.sess.run(self.sync_parameter)

            s_batch = []
            a_batch = []
            r_history = []

            start_step = local_step
            done = False
            lr = None

            # エピソード実行
            while not done and local_step - start_step < self.batch_size:
                # 行動を選択
                action = self.model.take_action(self.sess, [obs])

                # RMSPropの学習率更新
                lr = self.global_server.scheduler.value()

                # 行動を実行
                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward

                # 観測，報酬，行動をバッチへと追加
                s_batch.append(obs)
                a_batch.append(action)
                r_history.append(reward)

                obs = next_obs

                # step数を進める
                global_step += 1
                local_step += 1
                episode_len += 1
            a_batch = np.int32(np.array(a_batch))
            s_batch = np.float32(np.array(s_batch))

            # advantage計算用のRを計算
            R = 0
            if done:
                obs = self.env.reset()
            else:
                # bootstrap from last state
                R = self.model.estimate_value(self.sess, [obs])

            # 報酬のバッチを構築
            batch_size = len(s_batch)
            r_batch = np.zeros(batch_size)
            for i in reversed(range(batch_size)):
                R = r_history[i] + self.discount_fact * R
                r_batch[i] = R

            # global shared parameterを更新
            self.sess.run(self.apply_grads,
                          feed_dict={self.A: a_batch,
                                     self.R: r_batch,
                                     self.model.s: s_batch,
                                     self.global_server.lr: lr
                                     }
                          )
            if done:
                print('thread: {0}, '
                      'global step: {1}, '
                      'local step: {2}, '
                      'total reward: {3} '
                      'episode len: {4} '
                      'learning rate: {5}'
                      .format(self.thread_id, global_step, local_step,
                              total_reward, episode_len, lr))
                total_reward = 0
                episode_len = 0
                episode_num += 1
