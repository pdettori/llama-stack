# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import MetaReferenceAgentsDispatcherImplConfig
from .agents import TurnJobsList, TurnJobItem, gen_turn_job_id_list_key


async def get_provider_impl(
    config: MetaReferenceAgentsDispatcherImplConfig, deps: Dict[Api, ProviderSpec]
):
    from .agents import MetaReferenceAgentsDispatcherImpl

    impl = MetaReferenceAgentsDispatcherImpl(
        config,
        deps[Api.inference],
        deps[Api.vector_io],
        deps[Api.safety],
        deps[Api.tool_runtime],
        deps[Api.tool_groups],
    )
    await impl.initialize()
    return impl
