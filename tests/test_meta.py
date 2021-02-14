import tahini.meta


def test___author__():
    assert isinstance(tahini.meta.__author__, str)


def test___version__():
    assert isinstance(tahini.meta.__version__, str)
