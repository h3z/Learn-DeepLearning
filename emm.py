
import matplotlib.pyplot as plt


def graph(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    epochs = range(len(loss_values))

    line1 = plt.plot(epochs, loss_values, label='loss')
    line2 = plt.plot(epochs, history_dict['val_loss'], label='val loss')
    plt.setp(line1, linewidth=1.0, marker='+', markersize=1.0)
    plt.setp(line2, linewidth=1.0, marker='4', markersize=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    line1 = plt.plot(epochs, history_dict['val_accuracy'], label='Validation/Test Accuracy')
    line2 = plt.plot(epochs, history_dict['accuracy'], label='Training Accuracy')
    plt.setp(line1, linewidth=1.0, marker='+', markersize=1.0)
    plt.setp(line2, linewidth=1.0, marker='4', markersize=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()
