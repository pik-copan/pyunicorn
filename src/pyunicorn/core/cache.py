# This file is part of pyunicorn.
# Copyright (C) 2008--2024 Jonathan F. Donges and pyunicorn authors
# URL: <https://www.pik-potsdam.de/members/donges/software-2/software>
# License: BSD (3-clause)
#
# Please acknowledge and cite the use of this software and its authors
# when results are used in publications or published elsewhere.
#
# You can use the following reference:
# J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
# L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
# and J. Kurths, "Unified functional network and nonlinear time series analysis
# for complex systems science: The pyunicorn package"

"""
Provides a mix-in class that manages an LRU cache of derived quantities
at the instance level, with declared dependencies on mutable attributes.
"""

from abc import ABC, abstractmethod
from functools import lru_cache, wraps
from inspect import getmembers, ismethod
from typing import Tuple, Optional
from collections.abc import Hashable


class Cached(ABC):
    """
    Mix-in class which manages an LRU cache for subclass methods that
    constitute derived quantities, based on `@functools.lru_cache()`.
    The cache is populated by calling decorated methods (with new arguments),
    has a bounded number of slots per method, and can be cleared manually.

    A subclasses should:
      - decorate derived quantity methods with `@Cached.method()`,
      - provide a method `__cache_state__() -> Tuple[Hashable,...]`, which is
        used by `Cached` to define the `__eq__()` and `__hash__()` methods
        used by `@functools.lru_cache()`.

    Class attributes which affect subsequently *defined* subclasses:
      - cache_enable: toggles caching globally
      - lru_params:   sets `@functools.lru_cache()` parameters

    NOTE:
    The intended behaviour is specified by `tests/test_core/test_cache.py`.
    """
    cache_enable = True
    lru_params = {"maxsize": 32, "typed": True}

    @abstractmethod
    def __cache_state__(self) -> Tuple[Hashable, ...]:
        """
        Hashable tuple of mutable object attributes, which will determine the
        instance identity for ALL cached method lookups in this class,
        *in addition* to the built-in object `id()`. Returning an empty tuple
        amounts to declaring the object immutable in general. Mutable
        dependencies that are specific to a method should instead be declared
        via `@Cached.method(attrs=(...))`.

        NOTE:
        A subclass is responsible for the consistency and cost of this state
        descriptor. For example, hashing a large array attribute may be
        circumvented by declaring it as a property, with a custom setter method
        that increments a dedicated mutation counter.
        """

    def __eq__(self, other):
        return (self is other) and (
            self.__cache_state__() == other.__cache_state__())

    def __hash__(self):
        return hash((id(self),) + self.__cache_state__())

    @classmethod
    def method(cls, name: Optional[str] = None,
               attrs: Optional[Tuple[str, ...]] = None):
        """
        Caching decorator based on `@functools.lru_cache()`.

        Cache entries for decorated methods are indexed by the combination of:
          - the object `id()`,
          - the object-level mutable instance attributes as declared by the
            subclass method `__cache_state__()`,
          - the method-level mutable instance attributes as declared by the
            optional decorator argument `attrs`, and
          - the argument pattern at the call site, including the ordering of
            named arguments.

        The decorated method provides several attributes of its own, as defined
        by `@functools.lru_cache()`, including:
          - `cache_clear()`: delete this method cache for ALL `cls` instances
          - `__wrapped__`: the original method

        :arg name: Optionally print a message at the first method invocation.
        :arg attrs: Optionally declare attribute names as dependencies.

        NOTE:
        The same reasoning about consistency and cost applies to the `attrs`
        argument as to the `__cache_state__()` method.
        """
        # evaluated at decorator instantiation
        assert attrs is None or (
            isinstance(attrs, tuple) and len(attrs) > 0
            and all(isinstance(a, str) for a in attrs))
        assert name is None or isinstance(name, str)

        def wrapper(f):
            # evaluated at decorator application (method definition)
            if not cls.cache_enable:
                return f
            else:
                def uncached(self, *args, **kwargs):
                    # evaluated at uncached method invocation
                    if (
                      name is not None and
                      getattr(self, "silence_level", 0) <= 1):
                        print(f"Calculating {name}...")
                    return f(self, *args, **kwargs)

                if attrs is None:
                    # leave call signature unchanged
                    wrapped = lru_cache(**cls.lru_params)(uncached)
                else:
                    # present `attrs` as arguments for cache lookup
                    @lru_cache(**cls.lru_params)
                    def cached(self, *args, **kwargs):
                        return uncached(self, *args[len(attrs):], **kwargs)

                    @wraps(cached, assigned=(
                        'cache_info', 'cache_clear', '__wrapped__'))
                    def wrapped(self, *args, **kwargs):
                        _attrs = (getattr(self, a) for a in attrs)
                        return cached(self, *_attrs, *args, **kwargs)

                # fully decorated method
                return wraps(f)(wrapped)
        return wrapper

    def cache_clear(self, prefix: Optional[str] = None):
        """
        *Delete* all method caches for ALL instances of `self.__class__`,
        and also recursively for any owned `Cached` instances listed in
        `self.__cache_state__()`. This is simply a loop over the
        `cache_clear()` methods of all cached methods.

        :arg prefix: Optionally restrict the deleted caches by method name.

        NOTE:
        Instead, *invalidating* method caches for a SINGLE instance is achieved
        by modifying the relevant subset of mutable attributes, see
        `@Cached.method()`.
        """
        for n, m in getmembers(self, predicate=ismethod):
            if prefix is None or n.startswith(prefix):
                if all(hasattr(m, p) for p in ["cache_clear", "__wrapped__"]):
                    m.cache_clear()
        for attr in self.__cache_state__():
            if isinstance(attr, Cached):
                attr.cache_clear(prefix=prefix)
