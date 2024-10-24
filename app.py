from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture('rtsp://admin:ctucl2021@@192.168.10.17:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
        <html>
            <head>
                <title>Video Streaming</title>
            </head>
            <body>
                <h1>Streaming de video en tiempo real</h1>
                <img src="/video_feed" width="640" height="480">
            </body>
        </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
