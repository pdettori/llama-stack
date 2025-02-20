# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging
import signal
import sys
from pathlib import Path

import nest_asyncio
import yaml
from aiohttp import web
from dotenv import load_dotenv, find_dotenv

from bullmq import Worker
from .config import (
    get_run_config_file_path,
    initialize_redis_store_from_config,
    initialize_kvstore_from_config,
    get_worker_config
)
from llama_stack import LlamaStackAsLibraryClient
from .logger import setup_logging
from .runner import JobHandler
from .telemetry import setup_telemetry
from .workers import run_workers, shutdown_workers, create_worker

load_dotenv(find_dotenv())  # read local .env file
logger = logging.getLogger(__name__)


async def create_shutdown_event():
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.info("Signal received, shutting down.")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return shutdown_event


async def create_workers(queue_name):
    runners = await run_workers([queue_name],queue_name)
    logger.info(f"Workers started for queue '{queue_name}'")

    async def stop_workers():
        await shutdown_workers(runners)
        logger.info("Workers shut down successfully.")

    return ([worker for worker, _ in runners], stop_workers)


async def create_web_app(workers: list[Worker], health_check_port: int):
    async def healthcheck(request: web.Request) -> web.Response:
        if all(worker.running for worker in workers):
            return web.Response(status=200)
        return web.Response(status=503)

    app = web.Application()
    app.add_routes([web.get("/health", healthcheck)])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", health_check_port)
    await site.start()
    logger.info("HTTP server started")

    async def stop_web_app():
        await runner.cleanup()
        logger.info("HTTP server shut down successfully.")

    return stop_web_app


async def create_library_client(
    config_path_or_template_name: str,
) -> LlamaStackAsLibraryClient:
    logger.info(f"Using config: {config_path_or_template_name}")
    client = LlamaStackAsLibraryClient(
        config_path_or_template_name=config_path_or_template_name
    )
    if not await client.async_client.initialize():
        logger.error("Llama stack not built properly")
        sys.exit(1)
    return client

async def main():
    setup_logging()
    setup_telemetry()
    ls_client = await create_library_client(
        config_path_or_template_name=get_run_config_file_path()
    )

    # Load the YAML configuration file
    config_file = Path(get_run_config_file_path())
    config_dict = yaml.safe_load(config_file.read_text())

    try:
        kvstore = await initialize_kvstore_from_config(config_dict)
        logger.info("KVStore initialized successfully.")
    except ValueError as e:
        logger.error(f"Error initializing KVStore: {e}")
        sys.exit(-1)

    try:
        redis_client = await initialize_redis_store_from_config(config_dict)
        logger.info("RedisStore initialized successfully.")
    except ValueError as e:
        logger.error(f"Error initializing RedisStore: {e}")
        sys.exit(-1)

    try:
        worker_config = await get_worker_config(config_dict)
        print(worker_config)
    except ValueError as e:
        logger.error(f"Error getting  worker config: {e}")
        sys.exit(-1)    

    # This is required because of event loop nesting
    nest_asyncio.apply()
    job_handler = JobHandler(ls_client, kvstore, redis_client)
    run_worker = create_worker(worker_config.queue_name, job_handler.handleTurn, {}, redis_client)

    shutdown_event = await create_shutdown_event()
    workers, stop_workers = await create_workers(worker_config.queue_name)
    stop_web_app = await create_web_app(workers, worker_config.health_check_port)

    await shutdown_event.wait()

    await stop_workers()
    await stop_web_app()


if __name__ == "__main__":
    asyncio.run(main())
