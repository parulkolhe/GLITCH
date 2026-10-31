"""Candidate CNN architectures for the controlled comparison in ml/experiments.py.

Every candidate shares the same input contract as the original model so the
game's preprocessing, the NumPy inference engine and calibration code work
unchanged:

    uint8 28x28x1 "bitmap" -> Rescaling(1/255) -> [training-only augmentation]
    -> feature extractor -> GlobalAveragePooling -> Dropout
    -> Dense(128, relu) "embedding" -> Dropout -> Dense(32, softmax) "class_probabilities"

Candidates
----------
baseline_cnn     The existing 3-block CNN from ml/train.py (91k params), retrained
                 under the shared protocol so the comparison is like-for-like.
compact_bn_cnn   Two 3x3 convs per block, Conv -> BN -> ReLU ordering, no conv bias
                 (BN supplies it), 32/64/128 channels, GAP head. Tests whether more
                 depth per resolution helps the animal confusions.
residual_cnn     Small ResNet-style network: stem + three stages of basic residual
                 blocks with strided-conv downsampling and 1x1 projections.
separable_cnn    compact_bn_cnn with depthwise-separable convs after the stem: the
                 efficiency candidate, for a much smaller and faster model.
"""

from __future__ import annotations


AUGMENT = {"translation": 0.08, "rotation": 0.045, "zoom": 0.08}

_STROKE_LAYER = None


def stroke_width_layer(tf):
    """Training-only random stroke-width change (+-1 px), identity at inference.

    Motivation: ml/robustness.py shows the baseline loses 21 points of accuracy
    when strokes are 1 px thicker and 75 points when 1 px thinner, while canvas
    drawings vary in pen width relative to drawing size. Each training image is
    eroded (1/4), dilated (1/4) or left unchanged (1/2), using 2x2 grey-scale
    max/min filters on the [0,1] image.
    """
    global _STROKE_LAYER
    if _STROKE_LAYER is not None:
        return _STROKE_LAYER

    @tf.keras.utils.register_keras_serializable(package="glitch")
    class RandomStrokeWidth(tf.keras.layers.Layer):
        def call(self, x, training=None):
            if not training:
                return x
            thick = tf.nn.max_pool2d(x, ksize=2, strides=1, padding="SAME")
            thin = -tf.nn.max_pool2d(-x, ksize=2, strides=1, padding="SAME")
            r = tf.random.uniform((tf.shape(x)[0], 1, 1, 1))
            return tf.where(r < 0.25, thin, tf.where(r < 0.5, thick, x))

        def compute_output_shape(self, input_shape):
            return input_shape

    _STROKE_LAYER = RandomStrokeWidth
    return RandomStrokeWidth


def _input_stack(tf, augment: bool, name_prefix: str = "augment_", stroke: bool = False):
    inputs = tf.keras.Input(shape=(28, 28, 1), dtype=tf.uint8, name="bitmap")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="normalize_0_1")(inputs)
    if augment:
        x = tf.keras.layers.RandomTranslation(AUGMENT["translation"], AUGMENT["translation"], fill_mode="constant",
                                              fill_value=0.0, name=name_prefix + "translation")(x)
        x = tf.keras.layers.RandomRotation(AUGMENT["rotation"], fill_mode="constant", fill_value=0.0,
                                           name=name_prefix + "rotation")(x)
        x = tf.keras.layers.RandomZoom((-AUGMENT["zoom"], AUGMENT["zoom"]), (-AUGMENT["zoom"], AUGMENT["zoom"]),
                                       fill_mode="constant", fill_value=0.0, name=name_prefix + "scale")(x)
    if stroke:
        x = stroke_width_layer(tf)(name=name_prefix + "stroke_width")(x)
    return inputs, x


def _head(tf, x, num_classes: int, l2: float, feature_dropout=0.3, embedding_dropout=0.25):
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(feature_dropout, name="dropout_features")(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg, name="embedding")(x)
    x = tf.keras.layers.Dropout(embedding_dropout, name="dropout_embedding")(x)
    return tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)


def baseline_cnn(tf, num_classes: int, augment: bool = True, l2: float = 0.0, stroke: bool = False):
    inputs, x = _input_stack(tf, augment, stroke=stroke)
    for index, filters in enumerate((32, 64), start=1):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"block{index}_conv")(x)
        x = tf.keras.layers.BatchNormalization(name=f"block{index}_bn")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"block{index}_pool")(x)
    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu", name="block3_conv")(x)
    return tf.keras.Model(inputs, _head(tf, x, num_classes, l2), name="baseline_cnn")


def _conv_bn_relu(tf, x, filters, name, l2, strides=1, separable=False):
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    if separable:
        x = tf.keras.layers.SeparableConv2D(filters, 3, strides=strides, padding="same", use_bias=False,
                                            depthwise_regularizer=reg, pointwise_regularizer=reg, name=name + "_conv")(x)
    else:
        x = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False,
                                   kernel_regularizer=reg, name=name + "_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    return tf.keras.layers.ReLU(name=name + "_relu")(x)


def compact_bn_cnn(tf, num_classes: int, augment: bool = True, l2: float = 0.0, separable: bool = False, stroke: bool = False):
    inputs, x = _input_stack(tf, augment, stroke=stroke)
    for block, filters in enumerate((32, 64, 128), start=1):
        # the stem convolution stays regular even in the separable variant (1 input channel)
        x = _conv_bn_relu(tf, x, filters, f"block{block}a", l2, separable=separable and block > 1)
        x = _conv_bn_relu(tf, x, filters, f"block{block}b", l2, separable=separable)
        if block < 3:
            x = tf.keras.layers.MaxPooling2D(name=f"block{block}_pool")(x)
    name = "separable_cnn" if separable else "compact_bn_cnn"
    return tf.keras.Model(inputs, _head(tf, x, num_classes, l2), name=name)


def separable_cnn(tf, num_classes: int, augment: bool = True, l2: float = 0.0, stroke: bool = False):
    return compact_bn_cnn(tf, num_classes, augment, l2, separable=True, stroke=stroke)


def residual_cnn(tf, num_classes: int, augment: bool = True, l2: float = 0.0, stroke: bool = False):
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    inputs, x = _input_stack(tf, augment, stroke=stroke)
    x = _conv_bn_relu(tf, x, 32, "stem", l2)

    def block(x, filters, strides, name):
        shortcut = x
        y = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False,
                                   kernel_regularizer=reg, name=name + "_conv1")(x)
        y = tf.keras.layers.BatchNormalization(name=name + "_bn1")(y)
        y = tf.keras.layers.ReLU(name=name + "_relu1")(y)
        y = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg,
                                   name=name + "_conv2")(y)
        y = tf.keras.layers.BatchNormalization(name=name + "_bn2")(y)
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False,
                                              kernel_regularizer=reg, name=name + "_proj")(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(name=name + "_proj_bn")(shortcut)
        y = tf.keras.layers.Add(name=name + "_add")([y, shortcut])
        return tf.keras.layers.ReLU(name=name + "_out")(y)

    x = block(x, 32, 1, "stage1")
    x = block(x, 64, 2, "stage2")    # 14x14
    x = block(x, 128, 2, "stage3")   # 7x7
    return tf.keras.Model(inputs, _head(tf, x, num_classes, l2), name="residual_cnn")


CANDIDATES = {
    "baseline_cnn": baseline_cnn,
    "compact_bn_cnn": compact_bn_cnn,
    "residual_cnn": residual_cnn,
    "separable_cnn": separable_cnn,
}
