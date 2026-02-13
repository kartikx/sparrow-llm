import argparse
import logging
import uvicorn

from app.io_model import GenerateRequest
from core.engine import Engine
from core.utils import setup_logging
from app.api_server import app

logger = logging.getLogger(__name__)

def main(args):
    setup_logging(args.log_level)
    
    engine = Engine(args)
    
    app.state.engine = engine
    
    # uvicorn.run(app, host="0.0.0.0", port=args.port)

    output = engine.generate_output(GenerateRequest(prompt="how are you today?", max_tokens=10))
    print("==========")
    print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--log-level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    args = parser.parse_args()

    main(args)
    
