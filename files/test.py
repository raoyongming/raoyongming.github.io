import os
import json
import base64
import asyncio

from urllib.parse import (
    urlsplit,
    urlunsplit,
)

import httpx

from websockets.asyncio.client import (
    connect,
)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

HF_TOKEN = os.environ["HF_TOKEN"]

RELAY_KEY = os.environ["RELAY_KEY"]


HF_WS_URL = os.getenv(
    "HF_WS_URL",
    (
        "wss://"
        "raoyongming-model-test"
        ".hf.space/worker"
    ),
)


LOCAL_MCP_URL = os.getenv(
    "LOCAL_MCP_URL",
    "http://127.0.0.1:8731/mcp",
)


# 如果 localhost MCP 也需要 Bearer Token，
# 可以设置这个环境变量。
#
# 不需要就留空。
LOCAL_MCP_BEARER = os.getenv(
    "LOCAL_MCP_BEARER"
)


RECONNECT_DELAY = float(
    os.getenv(
        "RECONNECT_DELAY",
        "3",
    )
)


# ---------------------------------------------------------
# URL helper
# ---------------------------------------------------------

def with_query(
    url: str,
    query: str,
) -> str:

    if not query:
        return url

    parts = urlsplit(url)

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            query,
            parts.fragment,
        )
    )


# ---------------------------------------------------------
# Worker
# ---------------------------------------------------------

class RelayWorker:

    def __init__(self):

        self.send_lock = asyncio.Lock()

        # request id -> asyncio.Task
        self.tasks: dict[
            str,
            asyncio.Task,
        ] = {}

        # -------------------------------------------------
        # 这里非常重要：
        #
        # localhost MCP 请求不能经过
        # 公司 HTTP_PROXY。
        # -------------------------------------------------

        self.http = httpx.AsyncClient(
            trust_env=False,

            timeout=httpx.Timeout(
                connect=10.0,

                # MCP / render 可能非常慢。
                read=None,

                write=60.0,
                pool=60.0,
            ),

            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )


    # -----------------------------------------------------
    # Safe websocket send
    # -----------------------------------------------------

    async def send_json(
        self,
        ws,
        message: dict,
    ):

        async with self.send_lock:

            await ws.send(
                json.dumps(message)
            )


    # -----------------------------------------------------
    # One MCP HTTP request
    # -----------------------------------------------------

    async def handle_http_request(
        self,
        ws,
        message: dict,
    ):

        request_id = message["id"]

        method = (
            message["method"]
            .upper()
        )

        query = message.get(
            "query",
            "",
        )

        url = with_query(
            LOCAL_MCP_URL,
            query,
        )

        # -------------------------------------------------
        # MCP protocol headers
        # -------------------------------------------------

        headers = {
            k: v
            for k, v
            in message.get(
                "headers",
                {},
            ).items()
            if k.lower()
            in {
                "accept",
                "content-type",
                "mcp-protocol-version",
                "mcp-session-id",
                "last-event-id",
            }
        }

        # HF_TOKEN 永远不能转给 localhost。
        headers.pop(
            "authorization",
            None,
        )

        # localhost MCP 如果需要 auth。
        if LOCAL_MCP_BEARER:

            headers[
                "authorization"
            ] = (
                "Bearer "
                + LOCAL_MCP_BEARER
            )

        body = base64.b64decode(
            message.get(
                "body_b64",
                "",
            )
        )

        try:

            # ---------------------------------------------
            # 真正调用 Windows localhost MCP
            # ---------------------------------------------

            async with self.http.stream(
                method,
                url,
                headers=headers,
                content=(
                    body
                    if body
                    else None
                ),
            ) as response:

                # -----------------------------------------
                # HTTP response headers
                # -----------------------------------------

                response_headers = {
                    name.lower(): value
                    for name, value
                    in response.headers.items()
                    if name.lower()
                    in {
                        "content-type",
                        "mcp-session-id",
                        "cache-control",
                        "retry-after",
                        "www-authenticate",
                        "allow",
                    }
                }

                await self.send_json(
                    ws,
                    {
                        "type": "response_start",
                        "id": request_id,
                        "status_code": (
                            response.status_code
                        ),
                        "headers": (
                            response_headers
                        ),
                    },
                )

                # -----------------------------------------
                # Body / SSE streaming
                # -----------------------------------------

                async for chunk \
                        in response.aiter_bytes():

                    if not chunk:
                        continue

                    await self.send_json(
                        ws,
                        {
                            "type": "response_chunk",
                            "id": request_id,
                            "data_b64": (
                                base64.b64encode(
                                    chunk
                                ).decode(
                                    "ascii"
                                )
                            ),
                        },
                    )

                await self.send_json(
                    ws,
                    {
                        "type": "response_end",
                        "id": request_id,
                    },
                )

        except asyncio.CancelledError:

            # Remote MCP client disconnected.
            #
            # 离开 httpx.stream context 后，
            # localhost connection 会被关闭。
            raise

        except Exception as exc:

            try:
                await self.send_json(
                    ws,
                    {
                        "type": "response_error",
                        "id": request_id,
                        "error": repr(exc),
                    },
                )

            except Exception:
                pass


    # -----------------------------------------------------
    # Start request
    # -----------------------------------------------------

    def start_request(
        self,
        ws,
        message: dict,
    ):

        request_id = message["id"]

        old_task = self.tasks.get(
            request_id
        )

        if old_task is not None:
            old_task.cancel()

        task = asyncio.create_task(
            self.handle_http_request(
                ws,
                message,
            )
        )

        self.tasks[
            request_id
        ] = task

        def cleanup(
            _task,
            rid=request_id,
        ):
            self.tasks.pop(
                rid,
                None,
            )

        task.add_done_callback(
            cleanup
        )


    # -----------------------------------------------------
    # Cancel
    # -----------------------------------------------------

    def cancel_request(
        self,
        request_id: str,
    ):

        task = self.tasks.get(
            request_id
        )

        if task is not None:
            task.cancel()


    async def cancel_all(self):

        tasks = list(
            self.tasks.values()
        )

        for task in tasks:
            task.cancel()

        if tasks:

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        self.tasks.clear()


    # -----------------------------------------------------
    # Main connection
    # -----------------------------------------------------

    async def run(self):

        headers = {

            # HF Private Space authentication
            "Authorization": (
                f"Bearer {HF_TOKEN}"
            ),

            # Our relay authentication
            "X-Relay-Key": RELAY_KEY,
        }

        try:

            while True:

                try:

                    print(
                        f"Connecting to "
                        f"{HF_WS_URL}",
                        flush=True,
                    )

                    # websockets >= 15:
                    #
                    # proxy=True 会自动读取
                    # HTTP_PROXY / HTTPS_PROXY。
                    async with connect(
                        HF_WS_URL,

                        additional_headers=(
                            headers
                        ),

                        proxy=True,

                        ping_interval=20,
                        ping_timeout=20,

                        # MCP tool response
                        # 可能包含图片等较大数据。
                        max_size=None,

                        open_timeout=30,
                    ) as ws:

                        print(
                            "Connected to "
                            "HF MCP relay",
                            flush=True,
                        )

                        async for raw in ws:

                            message = (
                                json.loads(raw)
                            )

                            message_type = (
                                message.get(
                                    "type"
                                )
                            )

                            if (
                                message_type
                                == "http_request"
                            ):

                                self.start_request(
                                    ws,
                                    message,
                                )

                            elif (
                                message_type
                                == "cancel"
                            ):

                                self.cancel_request(
                                    message["id"]
                                )

                except Exception as exc:

                    print(
                        "Relay connection lost: "
                        f"{exc!r}",
                        flush=True,
                    )

                finally:

                    await self.cancel_all()

                await asyncio.sleep(
                    RECONNECT_DELAY
                )

        finally:

            await self.http.aclose()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

async def main():

    worker = RelayWorker()

    await worker.run()


if __name__ == "__main__":

    asyncio.run(main())
