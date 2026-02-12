import argparse
from engine import Engine
from utils import setup_logging
import logging
import uvicorn

from http_server import app

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--log-level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    engine = Engine(args)
    
    app.state.engine = engine
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
