# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import asyncio
from typing import List
from bullmq import Worker
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.exceptions import TimeoutError, ConnectionError
from redis.backoff import ExponentialBackoff

logger = logging.getLogger()


workers: dict[str, Worker] = dict()


def create_worker(queue_name: str, processor, opts, redis_client):
    worker = Worker(
        queue_name, processor, {**opts, "autorun": False, "connection": redis_client}
    )

    def completedCallback(job, result):
        logger.info("Job done")

    worker.on("completed", completedCallback)

    def failedCallback(job, err):
        logger.error("Job Failed", err)

    worker.on("failed", failedCallback)

    def errorCallback(err, job):
        logger.info("Worker failed", err)

    worker.on("error", errorCallback)

    workers[queue_name] = worker
    return worker


Runners = list[tuple[Worker, asyncio.Task]]


async def run_workers(names: List[str], queue):
    # TODO add autodiscovery
    from .runner import JobHandler

    tuples: Runners = []
    for name in names:
        worker = workers.get(name)
        if worker is not None:
            logger.info(f"Starting worker '{name}' on queue '{queue}'")
            task = asyncio.create_task(worker.run())
            tuples.append((worker, task))
    return tuples


async def shutdown_workers(runners: Runners):
    for worker, task in runners:
        await worker.close()
        await task
