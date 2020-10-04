from flask import Flask, render_template, request, send_from_directory, Response
from flask_socketio import SocketIO, send

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")

wsClients = []

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0")

@socketio.on('update requested')
def handle_message(message):
    print('received message: ' + message)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/image.jpg")
def return_image():
    # return send_from_directory("/dev/shm/", "output-7.jpg")
    return Response(image_data, mimetype="image/jpeg")

@socketio.on('connect')
def client_connected():
    wsClients.append(request)
    print(wsClients)

@socketio.on('disconnect')
def client_disconnected():
    wsClients.remove(request)
    print(wsClients)

@app.route('/request-update', methods=['POST'])
def requestUpdate():
    global image_data

    if wsClients:
        image_data = request.data
        socketio.send('Update is available.', json=False)
        return "Success."

    else:
        return "No client was notified because none was online."

# Inspiried by https://gist.github.com/ericremoreynolds/dbea9361a97179379f3b

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0")