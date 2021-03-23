@keras_export('keras.models.load_model')
def load_model(filepath, custom_objects=None, compile=True):  # pylint: disable=redefined-builtin
  """Loads a model saved via `save_model`.

  Arguments:
      filepath: One of the following:
          - String, path to the saved model
          - `h5py.File` object from which to load the model
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.
      compile: Boolean, whether to compile the model
          after loading.

  Returns:
      A Keras model instance. If an optimizer was found
      as part of the saved model, the model is already
      compiled. Otherwise, the model is uncompiled and
      a warning will be displayed. When `compile` is set
      to False, the compilation is omitted without any
      warning.

  Raises:
      ImportError: if loading from an hdf5 file and h5py is not available.
      IOError: In case of an invalid savefile.
  """
  if not tf2.enabled() or (
      h5py is not None and (
          isinstance(filepath, h5py.File) or h5py.is_hdf5(filepath))):
    return hdf5_format.load_model_from_hdf5(filepath, custom_objects, compile)

  if isinstance(filepath, six.string_types):
    loader_impl.parse_saved_model(filepath)
    return saved_model.load_from_saved_model(filepath)

  raise IOError(
      'Unable to load model. Filepath is not an hdf5 file (or h5py is not '
      'available) or SavedModel.')
