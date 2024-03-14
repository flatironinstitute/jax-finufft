import pytest
from jax_finufft import options


@pytest.mark.parametrize("opts", [None, options.Opts()])
def test_default_options(opts):
    assert options.unpack_opts(opts, 1, True) == opts
    assert options.unpack_opts(opts, 2, True) == opts
    assert options.unpack_opts(opts, 1, False) == opts
    assert options.unpack_opts(opts, 2, False) == opts


def test_nested_by_type():
    opts = options.NestedOpts(
        type1=options.Opts(spread_debug=True),
        type2=options.Opts(debug=True),
    )

    assert options.unpack_opts(opts, 1, True) == options.Opts(spread_debug=True)
    assert options.unpack_opts(opts, 2, True) == options.Opts(debug=True)


def test_nested_by_direction():
    opts = options.NestedOpts(
        forward=options.Opts(spread_debug=True),
        backward=options.Opts(debug=True),
    )

    assert options.unpack_opts(opts, 1, True) == options.Opts(spread_debug=True)
    assert options.unpack_opts(opts, 1, False) == options.Opts(debug=True)


def test_nested_multi():
    opts = options.NestedOpts(
        forward=options.Opts(spread_debug=True),
        backward=options.NestedOpts(
            type1=options.Opts(spread_debug=True, debug=True),
            type2=options.Opts(debug=True),
        ),
    )

    assert options.unpack_opts(opts, 1, True) == options.Opts(spread_debug=True)
    assert options.unpack_opts(opts, 1, False) == options.NestedOpts(
        type1=options.Opts(spread_debug=True, debug=True),
        type2=options.Opts(debug=True),
    )
    assert options.unpack_opts(
        options.unpack_opts(opts, 1, False), 1, True
    ) == options.Opts(spread_debug=True, debug=True)
    assert options.unpack_opts(
        options.unpack_opts(opts, 1, False), 2, True
    ) == options.Opts(debug=True)
