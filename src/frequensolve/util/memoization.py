"""
Memoization decorators

Adapted from Devito (under MIT license; see Devito for license): ::

   https://github.com/devitocodes/devito/blob/master/devito/tools/memoization.py

"""

from collections.abc import Hashable
from functools import partial
from itertools import tee

import numpy as np

__all__ = ["memoized_func", "memoized_meth", "memoized_generator", "quantize"]


class memoized_func:
    """Memoize calls to a standalone function with hashable arguments.

    Args:
        func: Function to wrap.
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kw):
        """Return a cached function result when arguments are hashable."""

        if not isinstance(args, Hashable):
            return self.func(*args, **kw)
        key = (self.func, args, frozenset(kw.items()))
        if key in self.cache:
            return self.cache[key]
        else:
            value = self.func(*args, **kw)
            self.cache[key] = value
            return value

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return partial(self.__call__, obj)


class memoized_meth:
    """Memoize instance-method calls on the owning object.

    Args:
        func: Method to wrap.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        """Support instance methods."""
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        """Return a cached method result stored on the owning object."""

        if not isinstance(args, Hashable):
            return self.func(*args)
        obj = args[0]
        try:
            cache = obj.__cache_meth
        except AttributeError:
            cache = obj.__cache_meth = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


class memoized_generator:
    """Memoize generator-producing methods while preserving independent iterators.

    Args:
        func: Generator method to wrap.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        """Support instance methods."""
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kwargs):
        """Return an independent iterator over cached generator output."""

        if not isinstance(args, Hashable):
            return self.func(*args)
        obj = args[0]
        try:
            cache = obj.__cache_gen
        except AttributeError:
            cache = obj.__cache_gen = {}
        key = (self.func, args[1:], frozenset(kwargs.items()))
        it = cache[key] if key in cache else self.func(*args, **kwargs)
        cache[key], result = tee(it)
        return result


class quantize:
    """Decorator that quantizes float arguments to nearest power of a base.

    Args:
        func: Decorated function.
    """

    def __init__(self, func):
        self.func = func
        self.base = 1.5

    def __call__(self, *args, **kwargs):
        """Return a wrapper that quantizes float positional arguments."""

        def wrapper(*args, **kwargs):
            args = list(args)
            base = kwargs["base"] if "base" in kwargs else self.base

            for i, arg in enumerate(args):
                if isinstance(arg, float):
                    power = round(np.log(arg) / np.log(base))
                    args[i] = self.base**power

            # Convert back to tuple
            args = tuple(args)

            return self.func(*args)

        return wrapper
