from pathlib import Path
from agentic_ai_analysis.core import local_server
import toml

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--path_toml_config", type=Path, required=True)
    args = parser.parse_args()
    config = toml.load(args.path_toml_config)

    _config_obj = local_server.LocalServerConfig(**config)
    local_server.start_local_server(_config_obj)

