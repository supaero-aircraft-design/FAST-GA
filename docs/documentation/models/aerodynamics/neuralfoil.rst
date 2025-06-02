################
Neuralfoil model
################

NeuralFoil :cite:`neuralfoil:2023` is a fast 2D airfoil analysis tool based on neural networks, offering XFoil-like
functionality without divergence issues and with reduced computation time. This package, used exclusively for airfoil
aerodynamic computations in both the ``highspeed`` and ``lowspeed`` aerodynamic submodels, is disabled by default in the
current version but may eventually replace XFoil in future updates.

.. code:: yaml

    model:
        aerodynamics_lowspeed:
            id: fastga.aerodynamics.lowspeed.legacy
            ⋮
            use_neuralfoil: false # true, false
        aerodynamics_highspeed:
            id: fastga.aerodynamics.highspeed.legacy
            ⋮
            use_neuralfoil: false # true, false

For detailed documentation on available functions and ML model training of NeuralFoil please check
`NeuralFoil Github Repository <https://github.com/peterdsharpe/NeuralFoil>`_.