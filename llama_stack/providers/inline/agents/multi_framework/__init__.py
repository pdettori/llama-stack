# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict
from pydantic import BaseModel

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import MultiFrameworkAgentImplConfig

from .math_agent import graph, MathAgent



async def get_provider_impl(
    config: MultiFrameworkAgentImplConfig, deps: Dict[Api, ProviderSpec]
):
    from .agents import MultiFrameworkAgentImpl

    impl = MultiFrameworkAgentImpl(
        config,
        deps[Api.inference],
        deps[Api.vector_io],
        deps[Api.safety],
        deps[Api.tool_runtime],
        deps[Api.tool_groups],
    )
    await impl.initialize()
    return impl


class MultiFrameworkAgentImplDataValidator(BaseModel):
    framework: str
    implementation_class: str