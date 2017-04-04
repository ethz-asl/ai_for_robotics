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
    return input / np.sum(input)


def compute_likelihood(map, measurement, prob_correct_measurement):
    likelihood = np.ones(len(map))
    likelihood[map == measurement] *= prob_correct_measurement
    likelihood[map != measurement] *= (1 - prob_correct_measurement)
    return likelihood


def measurement_update(prior, likelihood):
    posterior = prior * likelihood
    return normalize(posterior)


def prior_update(posterior, movement, movement_noise_kernel):
    kernel = movement_noise_kernel
    if movement < 0:
        kernel = np.flip(kernel, 0)
    return filters.convolve(np.roll(posterior, movement), kernel, mode='wrap')


def run_bayes_filter(measurements, motions, plot_histogram=False):
    map = np.array(
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0])
    sensor_prob_correct_measure = 0.9
    movement_noise_kernel = np.array([0.15, 0.8, 0.05])

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
