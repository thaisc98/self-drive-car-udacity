
import eventlet
import eventlet.websocket
import json

# Start WebSocket server
@eventlet.websocket.WebSocketWSGI
def handle(ws):
    sid = "simulator_sid" 
    print(f"Connected to simulator with sid: {sid}")
    ws.send('0{"sid":"%s","upgrades":[],"pingInterval":25000,"pingTimeout":20000,"maxPayload":1000000}' % sid)
    
    while True:
        msg = ws.wait()
        if msg is None:
            print("Connection closed")
            break
        print(f"Received message: {msg}")
        
        if msg == '2':
            ws.send('3')
        elif msg == '40':
            ws.send('40') 
            data = json.dumps(["steer", {'steering_angle': '0.1', 'throttle': '0.5'}])
            ws.send('42' + data)
            print(f"Sent initial steer: {data}")
        elif msg.startswith('42'): 
            json_str = msg[2:]
            try:
                event_data = json.loads(json_str)
                event = event_data[0]
                data = event_data[1] if len(event_data) > 1 else {}
                print(f"Received event: {event}, data: {data}")
                
                if event == "telemetry":
                    data = json.dumps(["steer", {'steering_angle': '0.1', 'throttle': '0.5'}])
                    ws.send('42' + data)
                    print(f"Sent steer: {data}")
                elif event == "manual":
                    data = json.dumps(["manual", {}])
                    ws.send('42' + data)
                    print(f"Sent manual: {data}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            print(f"Unknown message: {msg}")

if __name__ == '__main__':
    print("Starting WebSocket server...")
    listener = eventlet.listen(('', 4567))
    eventlet.wsgi.server(listener, handle)