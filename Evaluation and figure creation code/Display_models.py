# Import models from h5 file
from tensorflow.keras.models import load_model
lstm_tv = load_model('best_model/best_lstm_tv.h5')
lstm_st = load_model('best_model/best_lstm_st.h5')
gru_st = load_model('best_model/best_gru_st.h5')
gru_tv = load_model('best_model/best_gru_tv.h5')

from pathlib import Path

# Create a directory for saving model visualizations
output_dir = Path('model_visualizations')
output_dir.mkdir(exist_ok=True)

# Visualize the models
def visualize_model(model, model_name):
    """
    Visualize the model architecture.
    """
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=output_dir / f'{model_name}_architecture.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96)
    print(f'Model architecture saved as {model_name}_architecture.png')
visualize_model(lstm_tv, 'LSTM_TV')
visualize_model(lstm_st, 'LSTM_ST')
visualize_model(gru_st, 'GRU_ST')
visualize_model(gru_tv, 'GRU_TV')