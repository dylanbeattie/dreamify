import tensorflow as tf
import numpy as np
import PIL.Image

from timeit import default_timer as timer
# from tensorflow.keras.preprocessing import image


# Download an image and read it into a NumPy array.
def download(name):
    # image_path = tf.keras.utils.load_img(name)
    img = PIL.Image.open(name)
    return np.array(img)


# Normalize an image
def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


src_file = 'example.jpg'

original_img = download(src_file)
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


# deepdream = DeepDream(dream_model)
#
# def run_deep_dream_simple(img, steps=100, step_size=0.01):
#   # Convert from uint8 to the range expected by the model.
#   img = tf.keras.applications.inception_v3.preprocess_input(img)
#   img = tf.convert_to_tensor(img)
#   step_size = tf.convert_to_tensor(step_size)
#   steps_remaining = steps
#   step = 0
#   while steps_remaining:
#     if steps_remaining>100:
#       run_steps = tf.constant(100)
#     else:
#       run_steps = tf.constant(steps_remaining)
#     steps_remaining -= run_steps
#     step += run_steps
#
#     loss, img = deepdream(img, run_steps, tf.constant(step_size))
#     # show(deprocess(img))
#     print ("Step {}, loss {}".format(step, loss))
#
#
#   result = deprocess(img)
#   return result

def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return shift, img_rolled


shift, img_rolled = random_roll(np.array(original_img), 512)


class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, img, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[x:x + tile_size, y:y + tile_size]
                    loss = calc_loss(img_tile, self.model)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients


get_tiled_gradients = TiledGradients(dream_model)


def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01,
                                octaves=range(-2, 3), octave_scale=1.3):
    base_shape = tf.shape(img)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        # Scale the image based on the octave
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

    result = deprocess(img)
    return result


octaves = range(-2, 2)
# steps_per_octave = 100
step_size = 0.01
scale = 1.8

class Dream:
    octaves = range(-1, 1)
    steps_per_octave = 100
    step_size = 0.01
    scale = 1.5

    def __init__(self, octaves, steps_per_octave, step_size, scale):
        self.octaves = octaves
        self.steps_per_octave = steps_per_octave
        self.step_size = step_size
        self.scale = scale

    def __str__(self):
        return("octaves: {}, steps_per_octave: {}, step_size: {}, scale: {}".format(self.octaves, self.steps_per_octave, self.step_size, self.scale))


# dreams = [
#     Dream(range(-1, 2), 1, 0.01, 1.3),
#     Dream(range(-2, 2), 10, 0.01, 1.4),
#     Dream(range(-3, 2), 20, 0.01, 1.5),
#     Dream(range(-4, 2), 40, 0.01, 1.6),
#     Dream(range(-5, 2), 70, 0.01, 1.8),
# ]

dreams = [
    Dream(range(-1, 3), 1, 0.01, 1.3),
    Dream(range(-2, 3), 10, 0.01, 1.4),
    Dream(range(-2, 3), 20, 0.01, 1.5),
    Dream(range(-3, 3), 40, 0.01, 1.9),
    Dream(range(-3, 4), 60, 0.01, 2.2),
]

total_started_at = timer()
for idx, zob in enumerate(dreams):
    print(zob)
    started_at = timer()
    dream = run_deep_dream_with_octaves(original_img, octaves=zob.octaves, steps_per_octave=zob.steps_per_octave, octave_scale=zob.scale)
    tf.keras.preprocessing.image.save_img("{}_{}".format(idx, src_file), dream)
    now = timer()
    print("step {} took {} seconds".format(idx, now - started_at))

total_ended_at = timer()
print("TOTAL RUN took {} seconds".format(total_ended_at - total_started_at))