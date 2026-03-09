PARAM_GRID = {
    "hidden_channels": [32*i for i in range(1, 9)],
    "num_gc_layers": [i for i in range(1, 6)],
    "dropout": [0.0, 0.5],
    "use_jumping_knowledge": [True, False],
    "num_readout_layers": [i for i in range(0, 4)],
    "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2]
}

