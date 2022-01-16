class ObjectWrapper(object):
    """credits to the tqdm python package"""
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        return setattr(self._wrapped, name, value)

    def wrapper_getattr(self, name):
        """Actual `self.getattr` rather than self._wrapped.getattr"""
        try:
            return object.__getattr__(self, name)
        except AttributeError:  # py2
            return getattr(self, name)

    def wrapper_setattr(self, name, value):
        """Actual `self.setattr` rather than self._wrapped.setattr"""
        return object.__setattr__(self, name, value)

    def __init__(self, wrapped):
        """
        Thin wrapper around a given object
        """
        self.wrapper_setattr('_wrapped', wrapped)
