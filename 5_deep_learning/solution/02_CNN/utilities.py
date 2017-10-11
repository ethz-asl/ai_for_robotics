import matplotlib.pyplot as plt
import pickle


def load_dataset(path):
    with open(path, "rb") as input_file:
        data = pickle.load(input_file)
    print("Loaded dataset with " + str(data['images'].shape[0]) + " samples.")
    return data['images'], data['cls']


def plot_images(images, cls_names):

    assert len(images) == len(cls_names) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(
            images[i], cmap=plt.get_cmap('gray'), interpolation='spline16')

        # Name of the true class.
        xlabel = cls_names[i]

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
