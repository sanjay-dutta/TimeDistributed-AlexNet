import os
import tensorflow as tf
import numpy as np

def configure_gpu(memory_limit=8192):
    """Configure GPU memory growth to avoid CUDA OOM errors."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(f"Using {len(logical_gpus)} logical GPU(s)")
        except RuntimeError as e:
            print("GPU configuration failed:", e)

def set_global_seeds(seed=27):
    np.random.seed(seed)
    tf.random.set_seed(seed)
