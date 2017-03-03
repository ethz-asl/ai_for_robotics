#!/usr/bin/env python

import NoiseModel as noise
import Features as features
import DataGenerator as data

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")


n_samples = 10000
data_generator = data.DataGenerator()
input_data, output_data = data_generator.sample(n_samples)
data_generator.save()

