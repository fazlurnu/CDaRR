import pytest


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False,
                     help="render matplotlib figures during tests")


def pytest_configure(config):
    if not config.getoption("--plot", default=False):
        import matplotlib
        matplotlib.use("Agg")


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture(scope="session", autouse=True)
def bluesky_init():
    import bluesky as bs
    if not getattr(bs, "_joblib_inited", False):
        bs.init(mode="sim", detached=True)
        bs._joblib_inited = True
    return bs
