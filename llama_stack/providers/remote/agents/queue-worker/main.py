# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from logger import setup_logging
from telemetry import setup_telemetry
import asyncio
import signal
from aiohttp import web
from bullmq import Worker

from workers import run_workers, shutdown_workers, create_worker
from runner import JobHandler

from config import config
from llama_stack import LlamaStackAsLibraryClient

from llama_stack import LlamaStackAsLibraryClient
from dotenv import load_dotenv, find_dotenv
import nest_asyncio

_ = load_dotenv(find_dotenv()) # read local .env file
logger = logging.getLogger()

async def create_shudown_event():
    shutdown_event = asyncio.Event()

    def signal_handler(signal, frame):
        logger.info("Signal received, shutting down.")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return shutdown_event


async def create_workers():
    runners = await run_workers(config.run_bullmq_workers)
    logger.info("Workers started")

    async def stop_workers():
        await shutdown_workers(runners)
        logger.info("Workers shut down successfully.")
    return ([worker for worker, _ in runners], stop_workers)


async def create_web_app(workers: list[Worker]):
    async def healthcheck(request):
        healthy = True
        for worker in workers:
            if not worker.running:
                healthy = False
        if healthy:
            return web.Response(status=200)
        else:
            return web.Response(status=503)

    app = web.Application()
    app.add_routes([web.get('/health', healthcheck)])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', config.port)
    await site.start()
    logger.info("HTTP server started")

    async def stop_web_app():
        await runner.cleanup()
        logger.info("HTTP server shut down successfully.")
    return stop_web_app

async def create_library_client(config_path_or_template_name):
    print(f"using config: {config_path_or_template_name}")
    client = LlamaStackAsLibraryClient(config_path_or_template_name=config_path_or_template_name)
    if not await client.async_client.initialize():
        print("llama stack not built properly")
        sys.exit(1)
    return client

# For the time being, let's assume the run.yaml file is colocated with this code
# TODO - find better approach, perhaps specify in config file
def get_run_file_path():
    current_directory = os.path.dirname(__file__)
    return os.path.join(current_directory, "run.yaml")

async def main():
    setup_logging()
    setup_telemetry()
    ls_client = await create_library_client(config_path_or_template_name=get_run_file_path())

    nest_asyncio.apply()
    jobHandler = JobHandler(ls_client)
    runWorker = create_worker(config.bullmq_workers_queue, jobHandler.handleRun, {})

    shutdown_event = await create_shudown_event()
    workers, stop_workers = await create_workers()
    stop_web_app = await create_web_app(workers)

    await shutdown_event.wait()

    await stop_workers()
    await stop_web_app()

if __name__ == "__main__":
    asyncio.run(main())
