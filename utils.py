import matplotlib.pyplot as plt
import os

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def plot_loss(loss_values, filename=None):
    epochs = list(range(1, len(loss_values) + 1))
    
    plt.plot(epochs, loss_values, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Training Epochs')
    plt.grid(True)
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
