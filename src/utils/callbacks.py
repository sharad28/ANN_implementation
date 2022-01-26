import tensorflow as tf
import os
import numpy as np
import time
from src.utils.common import get_unique_filename

# def get_timestamp(name):
#     timestamp=time.asctime().replace(':',"_").replace(" ","_")
#     unique_name = f"{name}_at_{timestamp}"
#     return unique_name

def get_callbacks(config, X_train):
    logs = config['logs']
    unique_dir_name = get_unique_filename('tb_logs')
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs['logs_dir'],logs['TENSORBOARD_ROOT_LOG_DIR'], unique_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR,exist_ok = True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)

    file_writer = tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)

    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1))
        tf.summary.image('20 handwriten digit sample',
                         images, max_outputs=25, step=0)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=config['params']['patience'],
                                                         restore_best_weights=config['params']['restore_best_weights'])

    artifacts = config['artifacts']
    CKPT_dir = os.path.join(artifacts['artifacts_dir'],artifacts['CHECKPOINT_DIR'])
    os.makedirs(CKPT_dir, exist_ok=True)
    CKPT_path = os.path.join(CKPT_dir,'model_ckpt.h5')

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    return [tensorboard_cb,early_stopping_cb,ckpt_cb]