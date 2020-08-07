import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from functools import partial


class Trainer:
    def __init__(self, generator, discriminator, cycle_loss_weight, identity_loss_weight, gradient_penalty_weight,
                 learning_rate=PiecewiseConstantDecay(boundaries=[100], values=[2e-4, 2e-5]), beta_1=0.5):
        self.A2B_G = generator
        self.B2A_G = generator
        self.A_D = discriminator
        self.B_D = discriminator

        self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)

        self.cycle_loss_weight = cycle_loss_weight
        self.identity_loss_weight = identity_loss_weight
        self.gradient_penalty_weight = gradient_penalty_weight

        self.mean_squared_error = MeanSquaredError()
        self.mean_absolute_error = MeanAbsoluteError()

    def train(self, train_dataset, steps=200):
        step = 0

        for A, B in train_dataset.take(steps):
            step += 1

            A2B, B2A, g_loss_dict = self.train_G_step(A, B)
            d_loss_dict = self.train_D_step(A, B, A2B, B2A)

            if step % 10 == 0:
                print(f'{step}/{steps}, '
                      f'generator loss = {g_loss_dict["G_loss"]:.4f}, '
                      f'discriminator loss = {d_loss_dict["D_loss"]:.4f}')

    @tf.function
    def train_G_step(self, A, B):
        with tf.GradientTape() as g_tape:
            A = tf.cast(A, tf.float32)
            B = tf.cast(B, tf.float32)

            A2B = self.A2B_G(A, training=True)
            B2A = self.B2A_G(B, training=True)

            A2B2A = self.B2A_G(A2B, training=True)
            B2A2B = self.A2B_G(B2A, training=True)

            A2A = self.B2A_G(A, training=True)
            B2B = self.A2B_G(B, training=True)

            A2B_D_logits = self.B_D(A2B, training=True)
            B2A_D_logits = self.A_D(B2A, training=True)

            # loss
            A2B_G_loss = self._generator_loss(A2B_D_logits)
            B2A_G_loss = self._generator_loss(B2A_D_logits)

            A2B2A_cycle_loss = self._cycel_loss(A, A2B2A)
            B2A2B_cycle_loss = self._cycel_loss(B, B2A2B)

            A2A_identity_loss = self._identity_loss(A, A2A)
            B2B_identity_loss = self._identity_loss(B, B2B)

            G_loss = (A2B_G_loss + B2A_G_loss)
            G_loss += (A2B2A_cycle_loss + B2A2B_cycle_loss) * self.cycle_loss_weight
            G_loss += (A2A_identity_loss + B2B_identity_loss) * self.identity_loss_weight

        grad_of_generator = g_tape.gradient(G_loss, self.A2B_G.trainable_variables + self.B2A_G.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(grad_of_generator, self.A2B_G.trainable_variables + self.B2A_G.trainable_variables))

        return A2B, B2A, {'A2B_G_loss': A2B_G_loss, 'B2A_G_loss': B2A_G_loss,
                          'A2B2A_cycle_loss': A2B2A_cycle_loss, 'B2A2B_cycle_loss': B2A2B_cycle_loss,
                          'A2A_identity_loss': A2A_identity_loss, 'B2B_identity_loss': B2B_identity_loss,
                          'G_loss': G_loss}

    @tf.function
    def train_D_step(self, A, B, A2B, B2A):
        with tf.GradientTape() as d_tape:
            A_D_logits = self.A_D(A, training=True)
            B2A_D_logits = self.A_D(B2A, training=True)
            B_D_logits = self.B_D(B, training=True)
            A2B_D_logits = self.B_D(A2B, training=True)

            A_D_loss, B2A_D_loss = self._discriminator_loss(A_D_logits, B2A_D_logits)
            B_D_loss, A2B_D_loss = self._discriminator_loss(B_D_logits, A2B_D_logits)
            A_D_gp = self._gradient_penalty(partial(self.A_D, training=True), A, B2A)
            B_D_gp = self._gradient_penalty(partial(self.B_D, training=True), B, A2B)

            D_loss = A_D_loss + B2A_D_loss
            D_loss += B_D_loss + A2B_D_loss
            D_loss += (A_D_gp + B_D_gp) * self.gradient_penalty_weight

        grad_of_discriminator = d_tape.gradient(D_loss, self.A_D.trainable_variables + self.B_D.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(grad_of_discriminator, self.A_D.trainable_variables + self.B_D.trainable_variables))

        return {'A_D_loss': A_D_loss + B2A_D_loss,
                'B_D_loss': B_D_loss + A2B_D_loss,
                'D_loss': D_loss}

    def _generator_loss(self, f_logit):
        return self.mean_squared_error(tf.ones_like(f_logit), f_logit)

    def _discriminator_loss(self, r_logit, f_logit):
        r_loss = self.mean_squared_error(tf.ones_like(r_logit), r_logit)
        f_loss = self.mean_squared_error(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def _cycel_loss(self, a, a2b2a):
        return self.mean_absolute_error(a, a2b2a)

    def _identity_loss(self, a, a2a):
        return self.mean_absolute_error(a, a2a)

    @staticmethod
    def _gradient_penalty(f, real, fake):
        def _interpolate(a, b):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp