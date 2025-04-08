
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.initializers import GlorotNormal

class BaseModel:
    def __init__(self, input_shape, output_units, data_name, loss='nse'):
        self.input_shape = input_shape
        self.output_units = output_units
        self.loss = loss
        self.model = None
        self.data_name = data_name

    def build_model(self):
        pass

    def compile_model(self):
        self.model.compile(optimizer="RMSprop", loss=self.loss)

    def train_model(self, trainX, trainY, checkpoint_path, epochs=10, batch_size=16):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        tensorboard = TensorBoard(
            log_dir=f'logs/{self.data_name}',
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=False,
                              verbose=1,
                              save_freq='epoch')
            

        with tensorflow.device('/GPU:0'):
            history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stopping, tensorboard, cp_callback])
            return history
