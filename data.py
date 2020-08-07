import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE


class make_dataset:
    def __init__(self,
                 A_img_paths,
                 B_img_paths,
                 load_size,
                 crop_size):

        self.A_img_paths = A_img_paths
        self.B_img_paths = B_img_paths
        self.load_size = load_size
        self.crop_size = crop_size

    def dataset(self, batch_size, repeat_count=None, random_transform=False):
        A_dataset = self._images(self.A_img_paths)
        B_dataset = self._images(self.B_img_paths)

        dataset = tf.data.Dataset.zip((A_dataset, B_dataset))

        dataset = dataset.map(lambda a, b: img_transform(a, b, self.load_size, self.crop_size, random_transform),
                              num_parallel_calls=AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset

    @staticmethod
    def _images(img_paths):
        imgs = tf.data.Dataset.from_tensor_slices(img_paths)
        imgs = imgs.map(tf.io.read_file)
        imgs = imgs.map(lambda x: tf.image.decode_png(x, 3), num_parallel_calls=AUTOTUNE)
        return imgs


def img_transform(A_img, B_img, load_size, crop_size, random_transform):
    if random_transform:
        A_img = tf.image.random_flip_left_right(A_img)
        B_img = tf.image.random_flip_left_right(B_img)

    A_img = tf.image.resize(A_img, [load_size, load_size])
    B_img = tf.image.resize(B_img, [load_size, load_size])

    if random_transform:
        A_img = tf.image.random_crop(A_img, [crop_size, crop_size, tf.shape(A_img)[-1]])
        B_img = tf.image.random_crop(B_img, [crop_size, crop_size, tf.shape(B_img)[-1]])

    A_img = tf.clip_by_value(A_img, 0, 255) / 255.0
    B_img = tf.clip_by_value(B_img, 0, 255) / 255.0

    A_img = A_img * 2 - 1
    B_img = B_img * 2 - 1

    return A_img, B_img