import paho.mqtt.client as mqtt
import json

BROKER = "192.168.0.102"   # same as in ESP32 code
TOPIC = "sensor/tds_ph"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    print(f"TDS: {data['tds']} ppm | pH: {data['ph']}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, 1883, 60)
client.loop_forever()
