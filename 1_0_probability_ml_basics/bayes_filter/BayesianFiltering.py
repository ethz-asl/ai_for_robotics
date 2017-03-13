import numpy as np
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import time


def histogram_plot(pos, title=None, c='b'):
    axis = plt.gca()
    x = np.arange(len(pos))
    axis.bar(x, pos, color=c)
    plt.ylim((0, 1))
    plt.xticks(np.asarray(x) + 0.4, x)
    if title is not None:
        plt.title(title)


def normalize(input):
    normalized_value = np.ones(len(input))
    # TODO: Implement the normalization function
    return normalized_value


def compute_likelihood(map, measurement, prob_correct_measurement):
    likelihood = np.ones(len(map))
    # TODO: compute the likelihood
    return likelihood


def measurement_update(prior, likelihood):
    # TODO: compute posterior, use function normalize
    return np.ones(len(prior))  # TODO: change this line to return posterior


def prior_update(posterior, movement, movement_noise_kernel):
    # TODO: compute the new prior
    # HINT: be careful with the movement direction and noise kernel!
    return np.ones(
        len(posterior))  # TODO: change this line to return new prior


def run_bayes_filter(measurements, motions, plot_histogram=False):
    map = np.array([0] * 20)  # TODO: define the map
    sensor_prob_correct_measure = 1  # TODO: define the probability of correct measurement
    movement_noise_kernel = 1  # TODO: define noise kernel of the movement command

    # Assume uniform distribution since you do not know the starting position
    prior = np.array([1. / 20] * 20)
    likelihood = np.zeros(len(prior))

    number_of_iterations = len(measurements)
    if plot_histogram:
        fig = plt.figure("Bayes Filter")
    for iteration in range(number_of_iterations):
        # Compute the likelihood
        likelihood = compute_likelihood(map, measurements[iteration],
                                        sensor_prob_correct_measure)
        # Compute posterior
        posterior = measurement_update(prior, likelihood)
        if plot_histogram:
            plt.cla()
            histogram_plot(map, title="Measurement update", c='k')
            histogram_plot(posterior, title="Measurement update", c='y')
            fig.canvas.draw()
            plt.show(block=False)
            time.sleep(.5)

        # Update prior
        prior = prior_update(posterior, motions[iteration],
                             movement_noise_kernel)
        if plot_histogram:
            plt.cla()
            histogram_plot(map, title="Prior update", c='k')
            histogram_plot(prior, title="Prior update")
            fig.canvas.draw()
            plt.show(block=False)
            time.sleep(.5)
    plt.show()
    return prior
