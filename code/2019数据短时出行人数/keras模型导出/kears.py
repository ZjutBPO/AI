def load_model(filepath, custom_objects=None, compile=True):
    """Loads a model saved via `save_model`.

    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File or h5py.Group object from which to load the model
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.

    # Returns
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')
    model = None
    opened_new_file = not isinstance(filepath, h5py.Group)
    f = h5dict(filepath, 'r')
    try:
        model = _deserialize_model(f, custom_objects, compile)
    finally:
        if opened_new_file:
            f.close()
    return model
