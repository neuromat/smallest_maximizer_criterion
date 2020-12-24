import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from examples.simulation_study import simulation


def test_simulation_study():
    samples_path = os.path.abspath('./examples/simulation_study/samples')
    temp_folder = os.path.abspath('./examples/simulation_study/bb/tmp')
    results_folder = os.path.abspath('./examples/simulation_study/bb/results')

    simulation.run_simulation('model1',
                              temp_folder,
                              results_folder,
                              samples_path)

    simulation.run_simulation('model2',
                              temp_folder,
                              results_folder,
                              samples_path)
