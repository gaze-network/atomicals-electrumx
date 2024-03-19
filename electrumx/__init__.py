__version__ = "1.16.0"
__indexer_version__ = "gaze-electrumx-v0.1.0"
version = f'ElectrumX {__version__}'
version_short = __version__
indexer_version = __indexer_version__

from electrumx.server.controller import Controller
from electrumx.server.env import Env
