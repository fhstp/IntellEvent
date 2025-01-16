from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import onnxruntime as rt
import os
import sys

# Installer:
# pyinstaller --onefile --clean --hidden-import numpy --hidden-import onnxruntime --add-data "models/ic_intellevent.onnx;models" --add-data "models/fo_intellevent.onnx;models" vicon_server.py

providers = ['CPUExecutionProvider']

script_dir = os.path.dirname(os.path.abspath(__file__))
ic_model_path = os.path.join(script_dir, "models", "ic_intellevent.onnx")
fo_model_path = os.path.join(script_dir, "models", "fo_intellevent.onnx")

ic_model = rt.InferenceSession(ic_model_path, providers=providers)
fo_model = rt.InferenceSession(fo_model_path, providers=providers)


def load_config(config_file='config.json'):
    # Determine the base directory where the .exe or script is running
    if getattr(sys, 'frozen', False):  # If running as a PyInstaller bundle
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    # Look for the config file in the same directory as the executable
    config_path = os.path.join(base_path, config_file)

    try:
        with open(config_path, 'r') as file:
            print(f"Loading configuration from: {config_path}")  # Debug: Print path
            return json.load(file)
    except FileNotFoundError:
        print(f"Configuration file '{config_path}' not found. Using default settings.")
        return {"host": "127.0.0.1", "port": 5000}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{config_path}': {e}")
        return {"host": "127.0.0.1", "port": 5000}

config = load_config()


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Define routes
        if self.path == '/predict_ic':
            self.handle_predict(ic_model)
        elif self.path == '/predict_fo':
            self.handle_predict(fo_model)
        else:
            self.respond(404, {"error": "Route not found"})

    def handle_predict(self, model):
        try:
            # Read the length of the incoming data
            content_length = int(self.headers['Content-Length'])
            # Read and parse the incoming JSON data
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            # Ensure the necessary key exists in the request
            if 'traj' not in data:
                self.respond(400, {"error": "Missing 'traj' in request data"})
                return

            # Perform the prediction
            prediction = model.run(['time_distributed'], {"input_1": data['traj']})

            # Respond with the prediction as JSON
            self.respond(200, prediction[0].tolist())
        except Exception as e:
            self.respond(500, {"error": str(e)})

    def respond(self, status_code, data):
        # Send response code
        self.send_response(status_code)
        # Set headers
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        # Send JSON-encoded response
        self.wfile.write(json.dumps(data).encode())

def run(server_class=HTTPServer, handler_class=RequestHandler):
    server_address = (config['host'], config['port'])
    httpd = server_class(server_address, handler_class)
    print(f"Server started on {config['host']}:{config['port']}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()


