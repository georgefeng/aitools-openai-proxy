import logging
import os
import traceback

import httpx
# from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from httpx import NetworkError, TooManyRedirects, InvalidURL, ConnectTimeout, ReadTimeout, \
    RequestError, PoolTimeout
from starlette.background import BackgroundTask
from contextlib import asynccontextmanager
from typing import Union
from httpx import AsyncClient, Limits
import json

# from client_manager import ClientManager

# Load .env file
# load_dotenv()

# Read environment variables
# OPENAI_API_KEY = 'sk-XXXXXXXXXXXXXXXX'
OPENAI_API_BASE_URL = os.getenv('OPENAI_API_BASE_URL')
OPENAI_ORG = ""
VERBOSE_LOGGING = False
# Load and validate OPENAI_TIMEOUT
OPENAI_TIMEOUT = 60
OPENAI_TIMEOUT = 120

logger = logging.getLogger(__name__)

# Configure logging based on VERBOSE_LOGGING
if VERBOSE_LOGGING:
    # Configure logging with a specific format including a timestamp
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class ClientManager:
    def __init__(self, base_url, timeout=60, error_threshold=10):
        self.base_url = base_url
        self.timeout = timeout
        self.error_threshold = error_threshold
        self.error_counter = 0
        self.limits = Limits(max_connections=200, max_keepalive_connections=20)
        self.client: AsyncClient = AsyncClient(base_url=self.base_url, timeout=self.timeout, limits=self.limits)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized HTTP client with base URL {self.base_url} and timeout {self.timeout} and limits {self.limits}")

    async def reset_client(self):
        try:
            if self.client:
                await self.client.aclose()
        except Exception as e:
            self.logger.error(f"Error closing HTTP client during reset: {e}")
        finally:
            self.logger.info(f"Resetting HTTP client after {self.error_counter}/{self.error_threshold} errors seen.")
            self.client = AsyncClient(base_url=self.base_url, timeout=self.timeout)
            self.error_counter = 0
            self.logger.info(f"Reset HTTP client with base URL {self.base_url} and timeout {self.timeout} and limits {self.limits}")

    async def increment_error(self):
        self.error_counter += 1
        self.logger.error(f"Incrementing error counter to {self.error_counter}/{self.error_threshold}")
        if self.error_counter >= self.error_threshold:
            await self.reset_client()

    async def get_client(self):
        return self.client

    async def close(self):
        try:
            if self.client:
                await self.client.aclose()
        except Exception as e:
            self.logger.error(f"Error closing HTTP client: {e}")

app = FastAPI()
client_manager = ClientManager(base_url=OPENAI_API_BASE_URL, timeout=OPENAI_TIMEOUT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client_manager
    yield
    # Clean up the ML models and release the resources
    await client_manager.close()


async def clean_headers(headers: dict, api_key: str, org: str) -> dict:
    cleaned_headers = {k: v for k, v in headers.items() if k.lower() not in ['host', 'authorization']}
    cleaned_headers['Authorization'] = f'Bearer {api_key}'
    cleaned_headers['OpenAI-Organization'] = org
    return cleaned_headers

async def read_request_content(request: Request) -> str:
    body = b''
    async for chunk in request.stream():
        body += chunk
    return body.decode('utf-8')


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE","OPTIONS"])
async def proxy_openai(path: str, request: Request):
    """
        Proxy requests to the OpenAI API.

        :param path: The full path from the incoming request, including the 'openai/' prefix if present.
        :param request: The incoming request with headers, method, etc.
    """
    client = await client_manager.get_client()

    # Remove 'openai/' prefix from the path, if it exists
    api_path = path[len('openai/'):] if path.startswith('openai/') else path

    request_method = request.method
    request_headers = dict(request.headers)
    request_content = request.stream()


    # cleaned_headers = await clean_headers(request_headers, OPENAI_API_KEY, OPENAI_ORG)
    cleaned_headers = request_headers
    request_content = await read_request_content(request)

    try:
        messages = json.loads(request_content)
        # logger.info(messages)
        if messages['model'] == 'mymodel':
            messages['messages'][0]['content'] = 'My Content'
            messages['model'] = 'gpt-3.5-turbo'
        request_content = json.dumps(messages,ensure_ascii=False)
        # logger.info(json.loads(request_content)['messages'])

        del cleaned_headers['content-length']
    except:
        pass


    try:
        url = httpx.URL(path=api_path, query=request.url.query.encode("utf-8"))

        rp_req = client.build_request(request_method, url, timeout=OPENAI_TIMEOUT, headers=cleaned_headers,
                                      content=request_content)
        rp_resp = await client.send(rp_req, stream=True)
    except ReadTimeout as e:
        error_detail = f"Read Timeout [aitools] - Error: {e}"
        logger.error(f"ReadTimeout encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=408, detail=error_detail)
    except ConnectTimeout as e:
        await client_manager.increment_error()
        error_detail = f"Connect Timeout [aitools] - Error: {e}"
        logger.error(f"ConnectTimeout encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=408, detail=error_detail)
    except PoolTimeout as e:
        await client_manager.increment_error()
        error_detail = f"Service Unavailable due to Pool Timeout [aitools] - Error: {e}"
        logger.error(f"PoolTimeout encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=error_detail)
    except NetworkError as e:
        await client_manager.increment_error()
        error_detail = f"Service Unavailable [aitools] - Error: {e}"
        logger.error(f"NetworkError encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=error_detail)
    except TooManyRedirects as e:
        error_detail = f"Too Many Redirects [aitools] - Error: {e}"
        logger.error(f"TooManyRedirects encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=310, detail=error_detail)
    except InvalidURL as e:
        error_detail = f"Invalid URL [aitools] - Error: {e}"
        logger.error(f"InvalidURL encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=error_detail)
    except RequestError as e:
        await client_manager.increment_error()
        error_detail = f"Request Error [aitools] - Error: {e}"
        logger.error(f"RequestError encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)
    except Exception as e:
        await client_manager.increment_error()
        error_detail = f"Unknown Error [aitools] - Error: {e}"
        logger.error(f"Unknown Error encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

    if rp_resp.status_code != 200:
        logger.error(f"Non-200 status code from openai: {rp_resp.status_code}")

    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=rp_resp.headers,
        background=BackgroundTask(rp_resp.aclose)
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
