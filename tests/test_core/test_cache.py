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
Consistency checks for the method caching mix-in.
"""

import pytest
import numpy as np

from pyunicorn.core.cache import Cached


# pylint: disable=disallowed-name
class TestCached:

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
        # immutable instances
        X = cls.Foo()

        # wrapped method metadata
        methods = ["foo1", "foo2", "bar"]
        assert all(getattr(X, f).__doc__ == f.capitalize() for f in methods)

        for _ in range(3):
            # method calls
            X.cache_clear()
            m = Cached.lru_params["maxsize"]

            r1, o1 = [], []
            for _ in range(m + 1):
                r1.append(X.foo1(1))
                # pylint: disable-next=no-member
                o1.append(cls.Foo.foo1.__wrapped__(X, 1))
            r1.append(X.foo1(1.0))
            assert all(r == r1[0] for r in r1)
            assert all(r == o for r, o in zip(r1, o1))

            r2, o2 = [], []
            for i in range(2 * m):
                r2.append(X.foo2(*range(i)))
                # pylint: disable-next=no-member
                o2.append(cls.Foo.foo2.__wrapped__(X, *range(i)))
            assert (np.diff(r2) == range(2 * m - 1)).all()
            assert all(r == o for r, o in zip(r2, o2))

            r3, o3 = [], []
            for i in range(5):
                r3.append(X.bar())
                # pylint: disable-next=no-member
                o3.append(cls.Foo.bar.__wrapped__(X))
            assert all(r and isinstance(r, bool) for r in r3)
            assert all(r == o for r, o in zip(r3, o3))

            # cache lookups
            c1, c2, c3 = (getattr(X, f).cache_info() for f in methods)
            assert c1.maxsize == c2.maxsize == m
            assert (c1.currsize, c1.misses, c1.hits) == (2, 2, m)
            assert (c2.currsize, c2.misses, c2.hits) == (m, 2 * m, 0)
            assert (c3.currsize, c3.misses, c3.hits) == (1, 1, 4)

            # cache clearing
            X.cache_clear(prefix="foo")
            c1, c2, c3 = (getattr(X, f).cache_info() for f in methods)
            assert (c1.currsize, c1.misses, c1.hits) == (0, 0, 0)
            assert (c2.currsize, c2.misses, c2.hits) == (0, 0, 0)
            assert (c3.currsize, c3.misses, c3.hits) == (1, 1, 4)

            # logging behaviour
            capture = capfd.readouterr()
            assert capture.err == ""
            assert capture.out.split("\n")[:-1] == [
                "Calculating foo2..." for _ in range(2 * m)]

    @classmethod
    def test_instance_immutable(cls):
        """
        Dependence on immutable instance attributes.
        """
        # immutable instances
        X, Y = cls.Foo(), cls.Foo()
        m = Cached.lru_params["maxsize"]

        # method calls
        for i in range(m):
            X.foo1(i)
            Y.foo1(i)
        Y.foo1(m-1)

        # cache lookups
        x, y = (o.foo1.cache_info() for o in [X, Y])
        assert x == y
        assert (x.currsize, x.misses, x.hits) == (m, 2 * m, 1)

        # cache clearing
        X.cache_clear()
        x, y = (o.foo1.cache_info() for o in [X, Y])
        assert x == y
        assert (x.currsize, x.misses, x.hits) == (0, 0, 0)

    @classmethod
    def test_instance_mutable(cls):
        """
        Dependence on mutable instance attributes.
        """
        # mutable instances
        class Baz(cls.Bar):
            @Cached.method()
            def baz(self):
                self.counter += 1
                return self.counter

        X, Y = Baz(), Baz()

        # method calls
        m = Cached.lru_params["maxsize"]
        k, n = 3, m // 2
        assert k < n < m
        for _ in range(n):
            assert X.baz() == Y.baz()
        Y.counter = 0
        for _ in range(k):
            Y.baz()

        # cache lookups
        x, y = (o.baz.cache_info() for o in [X, Y])
        assert x == y
        assert (x.currsize, x.misses, x.hits) == (2 * n, 2 * n, k)

        # cache clearing
        X.cache_clear()
        x, y = (o.baz.cache_info() for o in [X, Y])
        assert x == y
        assert (x.currsize, x.misses, x.hits) == (0, 0, 0)

    @classmethod
    def test_instance_rec(cls):
        """
        Dependence on owned `Cached` instances.
        """
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
        m = Cached.lru_params["maxsize"]
        X = BarFoo(cls.Foo())
        for i in range(m):
            assert X.baz(i) == i
        assert X.counter == m
        for i in range(m):
            assert X.baz(i) == i
        assert X.counter == 2 * m

        # cache lookups
        x, y = (m.cache_info() for m in [X.baz, X.foo.foo1])
        assert (x.currsize, x.misses, x.hits) == (m, 2 * m, 0)
        assert (y.currsize, y.misses, y.hits) == (m, m, m)

        # cache clearing
        X.cache_clear()
        x, y = (m.cache_info() for m in [X.baz, X.foo.foo1])
        assert (x.currsize, x.misses, x.hits) == (0, 0, 0)
        assert (y.currsize, y.misses, y.hits) == (0, 0, 0)

    @classmethod
    def test_attributes(cls):
        """
        Dependence on method-specific attributes.
        """
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
        m = Cached.lru_params["maxsize"]
        k, n = 3, m // 2
        assert k < n < m
        for _ in range(n):
            X.baz()
            Y.baz()
        print()
        X.secret = 0
        Y.secret = 0
        for _ in range(n):
            X.baz()
            Y.baz()
        print()
        X.secret = 0
        Y.secret = 0
        Y.counter = 0
        for _ in range(k):
            X.baz()
            Y.baz()

        # cache lookups
        x1, y1 = (o.baz.cache_info() for o in [X, Y])
        assert (x1.currsize, x1.misses, x1.hits) == (n, n, n + k)
        assert (y1.currsize, y1.misses, y1.hits) == (2 * n, 2 * n, k)

        # cache clearing
        X.cache_clear()
        x2, y2 = (o.baz.cache_info() for o in [X, Y])
        assert (x2.currsize, x2.misses, x2.hits) == (0, 0, 0)
        assert y2 == y1
        Y.cache_clear()
        x3, y3 = (o.baz.cache_info() for o in [X, Y])
        assert x3 == x2
        assert (y3.currsize, y3.misses, y3.hits) == (0, 0, 0)

    @classmethod
    def test_disable(cls):
        """
        Dependence on global switch.
        """
        Cached.cache_enable = False

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
        m = Cached.lru_params["maxsize"]
        for _ in range(2 * m):
            X.baz()

        # no caching
        assert X.counter == 0
        assert X.undeclared_counter == 2 * m
        assert not hasattr(X.baz, "cache_info")

        # no-op
        X.cache_clear()
