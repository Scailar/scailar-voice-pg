from fastapi import FastAPI
from inbound_main import app as inbound_app
from outbound_live_main import app as outbound_app
from fastapi.middleware.cors import CORSMiddleware
import logging
from uvicorn.config import LOGGING_CONFIG


app = FastAPI()


# Merge routes so that endpoints from both apps are available on the same root
app.router.routes.extend(inbound_app.routes)
app.router.routes.extend(outbound_app.routes)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["*"] for all origins (not recommended in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# if __name__ == "__main__":
#     import uvicorn

#     log_config = uvicorn.config.LOGGING_CONFIG

#     log_config["formatters"]["error"][
#         "fmt"
#     ] = "%(asctime)s - %(levelname)s - %(message)s"
#     log_config["formatters"]["default"][
#         "fmt"
#     ] = "%(asctime)s - %(levelname)s - %(message)s"

#     uvicorn.run("main:app", host="0.0.0.0", port=5503, log_config=None)
