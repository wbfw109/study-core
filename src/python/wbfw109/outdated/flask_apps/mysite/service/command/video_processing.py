""" python Client of [python Client, flask Server, browser Client]

for test type in $bash
    : python backend/service/command/video_processing.py -m /home/wbfw109/image_processing/source/cutted_dog_video_1.mp4
    : python backend/service/command/video_processing.py -m /home/wbfw109/image_processing/source/dog_video_2.mp4
    : python backend/service/command/video_processing.py -m ALL

import socketio
    socketio.client(), socketio.Asyncclient()
        .emit()
            data - Data can be of type str, bytes, list or dict.

"""
from mysite.config import CONFIG_CLASS
from mysite.controller.stream.streaming import LocalVideos
import multiprocessing
from logging import error
from pathlib import Path
import argparse
import cv2
import asyncio
import socketio
import functools
import base64
import time
import os

if __name__ == "__main__":
    CONFIG_CLASS.ensure_folders_exist()

    # if await_sleep_second is too rapid, there is no data in the queue anyway. so it doesn't mean much.
    await_sleep_second = 0.05

    def put_frames_to_queue(
        arg_movie, local_videos: LocalVideos, queue: multiprocessing.Queue
    ):
        if arg_movie == "ALL":
            for video_path in local_videos.directory:
                for encoded_image in (
                    lambda: (yield from LocalVideos.generator_of_frame(video_path))
                )():
                    queue.put(base64.b64encode(encoded_image))
        elif arg_movie == "":
            for encoded_image in (
                lambda: (yield from LocalVideos.generator_of_frame(local_videos.index))
            )():
                queue.put(base64.b64encode(encoded_image))
        else:
            for encoded_image in (
                lambda: (yield from LocalVideos.generator_of_frame(local_videos.file))
            )():
                print("put")
                queue.put(base64.b64encode(encoded_image))

        # if this process ends, await_sleep_second set faster than previous.
        global await_sleep_second
        await_sleep_second = 0.03
        return True

    frame_queue = multiprocessing.Queue()

    # * parser Start
    parser = argparse.ArgumentParser(
        description="perform some operations for video processing.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v", "--verbosity", help="increase output verbosity", action="count", default=0
    )
    group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "-m",
        "--movie",
        help="specfied video file base directory is $home/image_processing/source. \ndefault value is PC cam (empty string) \n{ALL, '', <movie path>}",
        default="",
    )

    args = parser.parse_args()
    local_videos = LocalVideos()

    if args.movie == "ALL":
        local_videos = LocalVideos()
    elif args.movie == "":
        # if videos == camera device index
        local_videos = LocalVideos(index=0)
    else:
        # if videos == video file
        local_videos = LocalVideos(file=Path(args.movie))

    # * parser End
    # * Client socket Start
    socketio_client = socketio.AsyncClient()

    @socketio_client.on("connect", namespace="/origin")
    async def connect_from_origin():
        print("[socketio_client from origin] connection established")

    @socketio_client.on("disconnect", namespace="/origin")
    async def disconnect_from_origin():
        print("[socketio_client from origin] disconnected from server")

    @socketio_client.on("request_frame_from_origin", namespace="/origin")
    async def request_frame_from_origin(data):
        global await_sleep_second
        print(
            f"[socketio_client from origin] {data}. start to process that send frames"
        )
        # temporary solution for in async bug that 'socketio.exceptions.BadNamespaceError: /origin is not a connected namespace' in python-socketio"
        await socketio_client.sleep(1.0)

        while True:
            if not frame_queue.empty():
                # loop is blocking event. code to let other tasks get a share of the CPU.
                await socketio_client.sleep(await_sleep_second)
                await socketio_client.emit(
                    "response_frame_from_origin",
                    frame_queue.get(),
                    namespace="/origin",
                )
                print("emit")
            else:
                await socketio_client.sleep(await_sleep_second)

    async def start_client():
        await socketio_client.connect(
            "http://localhost:5000",
            transports=[
                "websocket",
            ],
        )
        await socketio_client.wait()

    # * Client socket End

    # * Process Start
    processes = [
        multiprocessing.Process(
            target=put_frames_to_queue, args=(args.movie, local_videos, frame_queue)
        ),
        multiprocessing.Process(target=asyncio.run, args=(start_client(),)),
    ]

    [p.start() for p in processes]
    # * Process End
