import socketio
import eventlet

sio = socketio.Server(async_mode='eventlet')
app = socketio.WSGIApp(sio)


@sio.on('connect')
def connect(sid, environ):
    print("SIM CONNECTED:", sid)
    sio.emit("steer", {"steering_angle": "0.0", "throttle": "0.2"}, to=sid)

@sio.on('telemetry')
def telemetry(sid, data):
    print("Telemetry received!")

eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
