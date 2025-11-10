from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import GlorotNormal
import tensorflow.keras.backend as K
import os,sys
import pickle

from model_defs.BaseModel import BaseModel
from model_defs.BasicLSTMModel import BasicLSTMModel
from model_defs.StackedLSTMModel import StackedLSTMModel
from model_defs.BidirectionalLSTMModel import BidirectionalLSTMModel
from model_defs.GRUModel import GRUModel
from tensorflow.keras.models import load_model


# Function to calculate NSE normalized squared error
def nseloss(y_true, y_pred):
  # Calculate the Normalized Squared Error (NSE) between predicted and true values
  return K.sum((y_pred-y_true)**2)/K.sum((y_true-K.mean(y_true))**2)

# Load training history from pickle file
def plot_model(model_name, valid_loss, data):

    # Load History
    with open(f'saved_model_multi/{data}/trainHistoryDict/{model_name}.pkl', "rb+") as file_pi:
        history = pickle.load(file_pi)

    # Create a plot for training and validation loss curves
    plt.figure()
    plt.title(f"{model_name} Training and Validation Loss")
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Embedded validation loss')
    plt.legend()

    # Display validation loss in the plot
    description = f"Validation set loss: {valid_loss}."
    plt.text(0.5, -0.1, description, transform=plt.gca().transAxes,
            fontsize=10, color='gray', ha='center', va='center')
    
    # Create directory to save the plot if it doesn't exist
    if not os.path.exists(f"lib/model_results/{data}/{model_name}"):
        os.makedirs(fr"lib/model_results/{data}/{model_name}") #If the saved model directory doesn't exist, make it    

    # Save the plot as a PNG file
    plt.savefig(fr"lib/model_results/{data}/{model_name}/{model_name}_training_validation.png")  
    plt.close() 



# Function to train ML models and save the trained history
def train_models(model_name, trainX, trainY, epochs=1, batch_size=16, loss="mse", load_models=False, data_name=None):
    if(load_models):
        return load_model(f'saved_model_multi/{data_name}/{model_name}_Saved_{data_name}')

    # Depending on the loss, we might need to pass a custom loss function
    if loss == "nse":
        loss = nseloss  

    model = None
    output_units = trainY.shape[1]

    # Instantiate the appropriate model class
    if(model_name == 'Basic_LSTM'):
        model = BasicLSTMModel(input_shape=(trainX.shape[1], trainX.shape[2]), output_units=output_units, loss=loss, data_name=data_name + "BasicLSTM")
    elif(model_name == 'Stacked_LSTM'):
        model = StackedLSTMModel(input_shape=(trainX.shape[1], trainX.shape[2]), output_units=output_units, loss=loss, data_name=data_name + "StackedLSTM")
    elif(model_name == 'Bidirectional_LSTM'):
        model = BidirectionalLSTMModel(input_shape=(trainX.shape[1], trainX.shape[2]), output_units=output_units, loss=loss, data_name=data_name + "BiDirectionalLSTM")
    elif(model_name == 'GRU'):
        model = GRUModel(input_shape=(trainX.shape[1], trainX.shape[2]), output_units=output_units, loss=loss, data_name=data_name + "GRU")

    # Save model and history
    model_directory = f"saved_model_multi/{data_name}"

    # Build and compile the model
    if model is not None:
        model.build_model()
    print(model.model.summary())
    # Train the model
    history = model.train_model(trainX, trainY, epochs=epochs, batch_size=batch_size, checkpoint_path= model_directory)
    
   
    print("saving model")
    model.model.save(f'{model_directory}\\{model_name}_Saved_{data_name}')    
    print("saving history") 

    os.makedirs(f"{model_directory}/trainHistoryDict", exist_ok=True) #If the saved model directory doesn't exist, make it    
    with open(f'{model_directory}/trainHistoryDict/{model_name}.pkl', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)

    return model.model


# Function to evaluate the model's validation loss
def evaluate_model(model, validX, validY):
    # Evaluate the model's performance on the validation set
    print("evaluating model")
    validation_loss = model.evaluate(validX, validY, verbose=1)
    print(f'Validation loss: {validation_loss}')
    return validation_loss


# Function to load a pre-trained model from a specified path
def get_model(model_name, data= 'Henry_2017_2020'):
    from keras.models import load_model

### TODO: FIX PATH - verify correct path structure
    path = f'saved_model_multi/{data}/{model_name}_Saved_{data}'
    if not os.path.exists(path):
        print(f"FILE NOT FOUND \n Given path: {path}")
        raise FileNotFoundError("Path to model does not exist. Check the dataname argument.")
    print(f"retrieving and loading {path} model.\n")
    
    try:
        model = load_model(path)
        print("model load status: SUCCESS")
        print(model.summary())

    except:
        print("model load status: FAILED")
        sys.exit()

    return model