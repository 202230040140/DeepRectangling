"""TensorFlow 1.x compatibility shims for running legacy DeepRectangling checkpoints."""

from __future__ import annotations

import sys
import types

import tensorflow as tf
import tf_slim


def enable_legacy_tf1() -> None:
    tf.compat.v1.disable_v2_behavior()

    contrib = types.ModuleType("tensorflow.contrib")
    contrib.slim = tf_slim
    contrib.layers = tf_slim.layers
    sys.modules["tensorflow.contrib"] = contrib
    sys.modules["tensorflow.contrib.slim"] = tf_slim
    sys.modules["tensorflow.contrib.layers"] = tf_slim.layers

    _RESIZE_METHODS = {
        0: tf.image.ResizeMethod.BILINEAR,
        1: tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        2: tf.image.ResizeMethod.BICUBIC,
        3: tf.image.ResizeMethod.AREA,
    }

    def resize_images(images, size, method=0):
        if isinstance(method, int):
            method = _RESIZE_METHODS.get(method, tf.image.ResizeMethod.BILINEAR)
        height, width = size if len(size) == 2 else (size[0], size[1])
        return tf.image.resize(images, (height, width), method=method)

    tf.resize_images = resize_images
    tf.image.resize_images = resize_images

    # Legacy model code uses tf.placeholder / tf.Session style names on tf directly.
    for name in (
        "placeholder",
        "Session",
        "ConfigProto",
        "global_variables_initializer",
        "train",
        "variable_scope",
        "name_scope",
        "get_variable_scope",
        "get_variable",
        "get_collection",
        "add_to_collection",
        "GraphKeys",
        "matrix_solve",
        "matrix_transpose",
        "matmul",
        "concat",
        "stack",
        "unstack",
        "reshape",
        "shape",
        "cast",
        "gather_nd",
        "clip_by_value",
        "floor",
        "meshgrid",
        "tile",
        "expand_dims",
        "squeeze",
        "reduce_sum",
        "reduce_mean",
        "reduce_max",
        "reduce_min",
        "stop_gradient",
        "gradients",
        "assign",
        "assign_add",
        "random_normal",
        "truncated_normal",
        "zeros",
        "ones",
        "zeros_like",
        "ones_like",
        "where",
        "equal",
        "not_equal",
        "greater",
        "less",
        "logical_and",
        "logical_or",
        "sigmoid",
        "tanh",
        "nn",
    ):
        if not hasattr(tf, name) and hasattr(tf.compat.v1, name):
            setattr(tf, name, getattr(tf.compat.v1, name))
