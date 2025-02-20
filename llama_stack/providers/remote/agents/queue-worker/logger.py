# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from .telemetry import logging_handler

LOG_LEVEL = "INFO"  # Use uppercase for recognized level

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG if LOG_LEVEL == 'TRACE' else getattr(logging, LOG_LEVEL), 
        handlers=[logging_handler, logging.StreamHandler()]
    )
