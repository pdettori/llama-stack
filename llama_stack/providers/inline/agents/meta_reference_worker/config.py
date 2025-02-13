# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel

from llama_stack.providers.utils.kvstore import KVStoreConfig
from llama_stack.providers.utils.kvstore.config import PostgresKVStoreConfig


class MetaReferenceAgentsWorkerImplConfig(BaseModel):
    persistence_store: KVStoreConfig

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> Dict[str, Any]:
        return {
            "persistence_store": PostgresKVStoreConfig.sample_run_config(
                db_name="llamastack_kvstore",
                type= "postgres",
                namespace=None,
                host="localhost",
                port=5432,
                db="postgres",
                user="postgres",
                password="mysecretpassword",
                table_name="llamastack_kvstore",
            )
        }
