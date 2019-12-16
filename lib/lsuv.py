import numpy as np
import tensorflow as tf


def svd_orthonormal(shape):
    # Orthonorm init code is taked from Lasagne
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


def get_activations(model, layer, X_batch):
    # Make intermediate model to get layer activation
    intermediate_layer_model = tf.keras.Model(
        inputs=model.get_input_at(0), outputs=layer.get_output_at(0))
    activations = intermediate_layer_model.predict(X_batch)
    return activations


def LSUVinitialize(model, X_batch, verbose=True, margin=0.1, max_iter=10):
    # only these layer classes considered for LSUV initialization;
    # add more if needed
    classes_to_consider = (tf.keras.layers.Dense, tf.keras.layers.Conv2D)

    needed_variance = 1.0
    layers_init = 0

    for layer in model.layers:
        if not isinstance(layer, classes_to_consider):
            continue
        # avoid small layers where activation variance close to zero,
        # esp. for small batches
        if np.prod(layer.get_output_shape_at(0)[1:]) < 32:
            print(layer.name, 'too small')
            continue
        print('Init Layer', layer.name)

        layers_init += 1
        weights_and_biases = layer.get_weights()
        # Set weights only with SVD init
        weights_and_biases[0] = svd_orthonormal(weights_and_biases[0].shape)
        layer.set_weights(weights_and_biases)
        # Get activations for the layer in inter
        activations = get_activations(model, layer, X_batch)
        # Variance of activations
        variance = np.std(activations)
        iteration = 0
        print(variance)
        while abs(needed_variance -
                  variance) > margin and iteration < max_iter:
            if np.abs(np.sqrt(variance)) < 1e-7:
                # avoid zero division
                break

            weights_and_biases = layer.get_weights()
            # Scale for weights only
            weights_and_biases[0] /= variance
            # weights_and_biases[1] -= mean
            layer.set_weights(weights_and_biases)
            activations = get_activations(model, layer, X_batch)
            variance = np.std(activations)

            iteration += 1

    print('LSUV: total layers initialized', layers_init)
    return model
