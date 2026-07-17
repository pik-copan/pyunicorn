# This file is part of pyunicorn.
# Copyright (C) 2008--2026 Jonathan F. Donges and pyunicorn authors
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
Consistency checks for the method caching mix-in.
"""

from gc import get_referrers
from weakref import getweakrefcount

import pytest
import numpy as np

from pyunicorn.core.cache import Cached


def check_wrefc(referent: object, count: int):
    """
    Check the number of weak references to `referent`.
    """
    assert getweakrefcount(referent) == count


def check_srefc(referent: object, count: int):
    """
    Check the number of strong references to `referent`.
    """
    assert len(get_referrers(referent)) == count


def check_single_sref(referent: object, obj: object, attr: str):
    """
    Check that there is one strong reference to `referent`, namely `obj.attr`.
    """
    assert getattr(obj, attr) is referent
    srefs = get_referrers(referent)
    assert len(srefs) == 1
    match srefs[0]:
        # either return type can occur, depending on runtime internals
        case dict() as r:
            assert r[attr] is referent
        case r:
            assert r is obj


class TestCached:

    # increase for testing purposes
    Cached.lru_params_local["maxsize"] = 8

    class Foo(Cached):
        silence_level = 1

        def __cache_state__(self):
            return ()

        @Cached.method()
        def foo1(self, a: int):
            """Foo1"""
            return a

        @Cached.method(name="foo2")
        def foo2(self, *args):
            """Foo2"""
            return sum(args, start=0)

        @Cached.method()
        def bar(self):
            """Bar"""
            return True

    class Bar(Cached):
        def __init__(self):
            self.counter = 0

        def __cache_state__(self):
            return (self.counter,)

    @classmethod
    def test_args(cls, capfd: pytest.CaptureFixture):
        """
        Dependence on method arguments.
        """
        gs = Cached.lru_params_global["maxsize"]
        ls = Cached.lru_params_local["maxsize"]
        b = 5

        # immutable instance
        X = cls.Foo()

        # wrapped method metadata
        methods = ["foo1", "foo2", "bar"]
        assert all(getattr(X, m).__doc__ == m.capitalize() for m in methods)

        # method calls
        for _ in range(3):
            X.cache_clear()

            for g in range(1, gs + 3):
                g_ = min(g, gs)
                X = cls.Foo()

                r1, o1 = [], []
                for _ in range(ls + 1):
                    r1.append(X.foo1(1))
                    o1.append(cls.Foo.foo1.__wrapped__(X, 1))
                r1.append(X.foo1(1.0))
                assert all(r == r1[0] for r in r1)
                assert all(r == o for r, o in zip(r1, o1))

                r2, o2 = [], []
                for i in range(2 * ls):
                    r2.append(X.foo2(*range(i)))
                    o2.append(cls.Foo.foo2.__wrapped__(X, *range(i)))
                assert (np.diff(r2) == range(2 * ls - 1)).all()
                assert all(r == o for r, o in zip(r2, o2))

                r3, o3 = [], []
                for i in range(b):
                    r3.append(X.bar())
                    o3.append(cls.Foo.bar.__wrapped__(X))
                assert all(r and isinstance(r, bool) for r in r3)
                assert all(r == o for r, o in zip(r3, o3))

                # global cache lookups
                gc1, gc2, gc3 = (getattr(X, m).cache_info() for m in methods)
                assert gc1.maxsize == gc2.maxsize == gs
                assert (gc1.currsize, gc1.misses, gc1.hits) == (
                    g_, g, g * (ls + 1))
                assert (gc2.currsize, gc2.misses, gc2.hits) == (
                    g_, g, g * (2 * ls - 1))
                assert (gc3.currsize, gc3.misses, gc3.hits) == (
                    g_, g, g * (b - 1))

                # local cache lookups
                lc1, lc2, lc3 = (
                    getattr(X, f"__cached_{m}__").cache_info()
                    for m in methods)
                assert lc1.maxsize == lc2.maxsize == lc3.maxsize == ls
                assert (lc1.currsize, lc1.misses, lc1.hits) == (2, 2, ls)
                assert (lc2.currsize, lc2.misses, lc2.hits) == (ls, 2 * ls, 0)
                assert (lc3.currsize, lc3.misses, lc3.hits) == (1, 1, 4)

            # cache clearing
            X.cache_clear(prefix="foo")
            gc1, gc2, gc3 = (getattr(X, m).cache_info() for m in methods)
            assert (gc1.currsize, gc1.misses, gc1.hits) == (0, 0, 0)
            assert (gc2.currsize, gc2.misses, gc2.hits) == (0, 0, 0)
            assert (gc3.currsize, gc3.misses, gc3.hits) == (gs, g, g * 4)

            # logging behaviour
            capture = capfd.readouterr()
            assert capture.err == ""
            msg2 = "Calculating foo2..."
            assert capture.out.split("\n")[:-1] == [msg2] * (g * 2 * ls)

    @classmethod
    def test_instance_immutable(cls):
        """
        Dependence on immutable instance attributes.
        """
        ls = Cached.lru_params_local["maxsize"]

        # immutable instances
        X, Y = cls.Foo(), cls.Foo()

        # method calls
        for i in range(ls):
            assert X.foo1(i) == Y.foo1(i) == i
        Y.foo1(ls-1)

        # global cache lookups
        gx, gy = (o.foo1.cache_info() for o in [X, Y])
        assert gx == gy
        assert (gx.currsize, gx.misses, gx.hits) == (2, 2, 2 * (ls - 1) + 1)

        # local cache lookups
        lx, ly = (o.__cached_foo1__.cache_info() for o in [X, Y])
        assert (lx.currsize, lx.misses, lx.hits) == (ls, ls, 0)
        assert (ly.currsize, ly.misses, ly.hits) == (ls, ls, 1)

        # cache clearing
        X.cache_clear()
        gx, gy = (o.foo1.cache_info() for o in [X, Y])
        assert gx == gy
        assert (gx.currsize, gx.misses, gx.hits) == (0, 0, 0)

    @classmethod
    def test_instance_mutable(cls):
        """
        Dependence on mutable instance attributes.
        """
        ls = Cached.lru_params_local["maxsize"]

        # mutable instances
        class Baz(cls.Bar):
            @Cached.method()
            def baz(self):
                self.counter += 1
                return self.counter

        X, Y = Baz(), Baz()

        # method calls
        k, n = 3, ls // 2
        assert k < n < ls
        for i in range(n):
            assert X.baz() == Y.baz() == i + 1
        Y.counter = 0
        for _ in range(k):
            Y.baz()

        # global cache lookups
        gx, gy = (o.baz.cache_info() for o in [X, Y])
        assert gx == gy
        assert (gx.currsize, gx.misses, gx.hits) == (2, 2, 2 * (n - 1) + k)

        # local cache lookups
        lx, ly = (o.__cached_baz__.cache_info() for o in [X, Y])
        assert (lx.currsize, lx.misses, lx.hits) == (n, n, 0)
        assert (ly.currsize, ly.misses, ly.hits) == (n, n, k)

        # cache clearing
        X.cache_clear()
        gx, gy = (o.baz.cache_info() for o in [X, Y])
        assert gx == gy
        assert (gx.currsize, gx.misses, gx.hits) == (0, 0, 0)

    @classmethod
    def test_instance_rec(cls):
        """
        Dependence on owned `Cached` instances.
        """
        ls = Cached.lru_params_local["maxsize"]

        # mutable instances
        class BarFoo(cls.Bar):
            def __init__(self, foo: cls.Foo):
                self.foo = foo
                cls.Bar.__init__(self)

            def __cache_state__(self):
                return cls.Bar.__cache_state__(self) + (self.foo,)

            @Cached.method()
            def baz(self, a: int):
                f = self.foo.foo1(a)
                self.counter += 1
                return f

        # method calls
        X = BarFoo(cls.Foo())
        for i in range(ls):
            assert X.baz(i) == i
        assert X.counter == ls
        for i in range(ls):
            assert X.baz(i) == i
        assert X.counter == 2 * ls

        # global cache lookups
        gx, gy = (m.cache_info() for m in [X.baz, X.foo.foo1])
        assert gx == gy
        assert (gx.currsize, gx.misses, gx.hits) == (1, 1, 2 * ls - 1)

        # local cache lookups
        lx, ly = (m.cache_info()
                  for m in [X.__cached_baz__, X.foo.__cached_foo1__])
        assert (lx.currsize, lx.misses, lx.hits) == (ls, 2 * ls, 0)
        assert (ly.currsize, ly.misses, ly.hits) == (ls, ls, ls)

        # cache clearing
        X.cache_clear()
        gx, gy = (m.cache_info() for m in [X.baz, X.foo.foo1])
        assert gx == gy
        assert (gx.currsize, gx.misses, gx.hits) == (0, 0, 0)

    @classmethod
    def test_attributes(cls):
        """
        Dependence on method-specific attributes.
        """
        ls = Cached.lru_params_local["maxsize"]

        # mutable instances
        class FooBaz(cls.Foo):
            def __init__(self):
                self.secret = 0

            @Cached.method(attrs=("secret",))
            def baz(self):
                """FooBaz"""
                self.secret += 1
                return self.secret

        class BarBaz(cls.Bar):
            def __init__(self):
                cls.Bar.__init__(self)
                self.secret = 0

            @Cached.method(attrs=("secret",))
            def baz(self):
                """BarBaz"""
                self.counter += 1
                self.secret += 1
                return self.counter + self.secret

        X, Y = FooBaz(), BarBaz()

        # wrapped method metadata
        assert all(o.baz.__doc__ == type(o).__name__ for o in [X, Y])

        # method calls
        k, n = 3, ls // 2
        assert k < n < ls
        for i in range(n):
            assert X.baz() == i + 1
            assert Y.baz() == 2 * (i + 1)
        X.secret = 0
        Y.secret = 0
        for i in range(n):
            assert X.baz() == 1
            assert Y.baz() == n + 2 * (i + 1)
        X.secret = 0
        Y.secret = 0
        Y.counter = 0
        for i in range(k):
            assert X.baz() == 1
            assert Y.baz() == 2

        # global cache lookups
        gx1, gy1 = (o.baz.cache_info() for o in [X, Y])
        assert (gx1.currsize, gx1.misses, gx1.hits) == (1, 1, 2 * n + k - 1)
        assert gy1 == gx1

        # local cache lookups
        lx1, ly1 = (o.__cached_baz__.cache_info() for o in [X, Y])
        assert (lx1.currsize, lx1.misses, lx1.hits) == (n, n, n + k)
        assert (ly1.currsize, ly1.misses, ly1.hits) == (2 * n, 2 * n, k)

        # cache clearing
        X.cache_clear()
        gx2, gy2 = (o.baz.cache_info() for o in [X, Y])
        assert (gx2.currsize, gx2.misses, gx2.hits) == (0, 0, 0)
        assert gy2 == gy1
        assert not hasattr(X, "__cached_baz__")
        ly2 = Y.__cached_baz__.cache_info()
        assert ly2 == ly1
        Y.cache_clear()
        gx3, gy3 = (o.baz.cache_info() for o in [X, Y])
        assert gx3 == gx2
        assert gy3 == gx3
        assert not hasattr(Y, "__cached_baz__")

    @classmethod
    def test_disable(cls):
        """
        Dependence on the global switch.
        """
        Cached.cache_enable = False
        ls = Cached.lru_params_local["maxsize"]

        class Baz(cls.Bar):
            def __init__(self):
                cls.Bar.__init__(self)
                self.undeclared_counter = 0

            @Cached.method()
            def baz(self):
                """Baz"""
                self.undeclared_counter += 1

        Cached.cache_enable = True
        X = Baz()

        # wrapped method metadata
        assert X.baz.__doc__ == "Baz"

        # method calls
        for _ in range(2 * ls):
            X.baz()

        # no caching
        assert X.counter == 0
        assert X.undeclared_counter == 2 * ls
        assert not hasattr(X.baz, "cache_info")

        # no-op
        X.cache_clear()

    @classmethod
    def test_refcount(cls):
        """
        Interaction with the garbage collector.
        """
        ls = Cached.lru_params_local["maxsize"]
        b = 5

        class FooBar(cls.Foo):
            def __init__(self, referent: cls.Bar):
                self.referent = referent

            @Cached.method()
            def bar(self) -> cls.Bar:
                return self.referent

            def __del__(self):
                self.referent.counter += 1

        # tracked instance
        Z = cls.Bar()
        X = FooBar(Z)
        check_single_sref(Z, X, "referent")
        check_wrefc(Z, 0)

        # global cache parametrisation
        methods = ["foo1", "foo2", "bar"]
        for m in methods:
            assert getattr(X, m).__wrapped__ is getattr(FooBar, m).__wrapped__
            assert getattr(X, m).cache_clear is getattr(FooBar, m).cache_clear
            assert getattr(X, m).cache_parameters() == Cached.lru_params_global

        # sequence designed to cover the case of dead local caches
        for del_instance in [False, True]:

            # empty cache
            r = 0
            check_srefc(X, 0)
            check_wrefc(X, r)

            # method calls
            for i in range(ls + b):
                if i == 0:
                    # expect new `ref(X)` in global cache
                    r += 1
                for m in methods:
                    match m:
                        case "foo1":
                            assert X.foo1(i) == i
                        case "foo2":
                            assert X.foo2() == 0
                        case "bar":
                            assert X.bar() is Z
                    check_srefc(Z, 1 if (i == 0 and not m == "bar") else 2)
                    check_wrefc(X, r)

            # populated global/local caches
            for m in methods:
                assert getattr(FooBar, m).cache_info().currsize == 1
                assert (getattr(X, f"__cached_{m}__").cache_info().currsize
                        == (ls if m == "foo1" else 1))
            check_srefc(X, 0)

            if not del_instance:
                # clear global and local caches, but keep instance
                X.cache_clear()
                check_srefc(X, 0)
                check_wrefc(X, 0)
                check_single_sref(Z, X, "referent")
                check_wrefc(Z, 0)
                assert Z.counter == 0
                for m in methods:
                    assert getattr(FooBar, m).cache_info().currsize == 0
            else:
                # delete instance, but keep global cache
                del X
                check_srefc(Z, 0)
                check_wrefc(Z, 0)
                assert Z.counter == 1
                for m in methods:
                    assert getattr(FooBar, m).cache_info().currsize == 1
