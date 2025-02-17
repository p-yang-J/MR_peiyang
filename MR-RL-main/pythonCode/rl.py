import numpy as np
import tensorflow as tf

#####################  hyper parameters  ####################

LR_A = 1    # learning rate for actor
LR_C = 1    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 3000
BATCH_SIZE = 32

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False

        self.a_dim, self.s_dim = a_dim, s_dim

        self.actor_model = self._build_a()
        self.critic_model = self._build_c()
        self.actor_target_model = self._build_a()
        self.critic_target_model = self._build_c()

        self.actor_optimizer = tf.keras.optimizers.Adam(LR_A)
        self.critic_optimizer = tf.keras.optimizers.Adam(LR_C)

        self.critic_loss = tf.keras.losses.MeanSquaredError()

    def _build_a(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(self.a_dim, activation='linear')
        ])
        return model

    def _build_c(self):
        s_input = tf.keras.layers.Input(shape=(self.s_dim,), name='s_input')
        a_input = tf.keras.layers.Input(shape=(self.a_dim,), name='a_input')
        s_dense = tf.keras.layers.Dense(30, activation='relu')(s_input)
        a_dense = tf.keras.layers.Dense(30, activation='relu')(a_input)

        net = tf.keras.layers.Add()([s_dense, a_dense])
        net = tf.keras.layers.Activation('relu')(net)
        output = tf.keras.layers.Dense(1)(net)

        model = tf.keras.models.Model(inputs=[s_input, a_input], outputs=output)
        return model

    def choose_action(self, s):
        return self.actor_model(s[None, :]).numpy()[0]

    @tf.function
    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        with tf.GradientTape() as tape:
            a_ = self.actor_target_model(bs_)
            q_ = self.critic_target_model([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic_model([bs, ba])
            critic_loss = self.critic_loss(y, q)

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad,self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            a = self.actor_model(bs)
            q = self.critic_model([bs, a])
            actor_loss = -tf.reduce_mean(q)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        self._soft_update(self.actor_target_model, self.actor_model)
        self._soft_update(self.critic_target_model, self.critic_model)

    def _soft_update(self, target_model, source_model):
        for target_param, param in zip(target_model.trainable_variables, source_model.trainable_variables):
            target_param.assign(TAU * param + (1 - TAU) * target_param)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:
            self.memory_full = True

    def save(self):
        self.actor_model.save_weights('actor_model')
        self.critic_model.save_weights('critic_model')

    def restore(self):
        self.actor_model.load_weights('actor_model')
        self.critic_model.load_weights('critic_model')



