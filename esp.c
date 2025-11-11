#include <WiFi.h>
#include <WebServer.h>
#include <PubSubClient.h>

// =========================
// 🔧 CONFIGURATION
// =========================

// Wi-Fi credentials
const char* ssid = "Phone";
const char* password = "ullasgss";

// MQTT credentials
const char* mqtt_server = "172.20.10.10"; // Replace with your Raspberry Pi IP address
const int mqtt_port = 1883;  // Default MQTT port

// MQTT client
WiFiClient espClient;
PubSubClient client(espClient);

// TDS Sensor
const int tdsPin = 33; // Analog input pin
const float VREF = 3.3; // Reference voltage for ESP32
const int ADC_RES = 4096; // 12-bit ADC resolution

// pH Sensor
#define PH_PIN 35 // ADC pin connected to pH sensor
float cal4_voltage = 1.7461;
float cal7_voltage = 1.8797;
float cal9_voltage = 2.039;

// Calibration values
float temperature = 25.0; // Default temperature (°C)
float K_VALUE = 0.60; // Calibrated constant (adjusted using 303 ppm solution)

// Calibration constant (adjust with known TDS sample)

// Variables
float voltage = 0.0;
float tdsValue = 0.0;
float lastVoltage = 0.0;
float lastPH = 0.0;
WebServer server(80);

// =========================
// 🧠 SETUP
// =========================
void setup() {
  Serial.begin(115200);
  delay(2000);

  // ADC setup
  analogReadResolution(12);
  analogSetPinAttenuation(tdsPin, ADC_11db); // Allows full 3.3V range
  analogSetPinAttenuation(PH_PIN, ADC_11db); // Measure up to ~3.3V reliably
  pinMode(tdsPin, INPUT);

  // Wi-Fi connection
  connectToWiFi();

  // MQTT connection
  client.setServer(mqtt_server, mqtt_port);
  
  // Web routes
  server.on("/", handleRoot);
  server.on("/tds", handleTDS);
  server.on("/data", handleData);
  server.begin();

  Serial.println("HTTP server started");
  Serial.println("\n=== ESP32 pH Sensor Reading (Pre-calibrated) ===");
  Serial.println("Using pre-set calibration values:");
  Serial.print("pH4: "); Serial.println(cal4_voltage, 4);
  Serial.print("pH7: "); Serial.println(cal7_voltage, 4);
  Serial.print("pH9: "); Serial.println(cal9_voltage, 4);
  Serial.println();
}

// =========================
// 🔁 LOOP
// =========================
void loop() {
  server.handleClient();
  static unsigned long lastRead = 0;
  if (millis() - lastRead >= 1000) {
    lastRead = millis();
    readTDSSensor();
    printSensorData();

    // Publish the data over MQTT
    publishSensorData();
  }
}

// =========================
// 📶 Wi-Fi Connect
// =========================
void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ Failed to connect to WiFi");
  }
}

// =========================
// 💧 TDS Reading Function
// =========================
void readTDSSensor() {
  const int samples = 30;
  int adcValue = 0;
  for (int i = 0; i < samples; i++) {
    adcValue += analogRead(tdsPin);
    delay(5);
  }
  adcValue /= samples;
  voltage = (adcValue / (float)ADC_RES) * VREF;
  float compensationCoefficient = 1.0 + 0.02 * (temperature - 25.0);
  float compensatedVoltage = voltage / compensationCoefficient;

  // Low-voltage fallback for low TDS
  if (compensatedVoltage < 0.1) {
    tdsValue = (compensatedVoltage / VREF) * 500 * K_VALUE; // tune 500 to match 34 ppm
  } else {
    tdsValue = (133.42 * compensatedVoltage * compensatedVoltage * compensatedVoltage
                - 255.86 * compensatedVoltage * compensatedVoltage
                + 857.39 * compensatedVoltage) * K_VALUE;
  }
  if (tdsValue < 0) tdsValue = 0;
}

// =========================
// 📊 pH Reading Function
// =========================
float readVoltage() {
  const int N = 30;
  int samples[N];
  for (int i = 0; i < N; i++) {
    samples[i] = analogRead(PH_PIN);
    delay(5);
  }

  // sort
  for (int i = 0; i < N - 1; i++) {
    for (int j = i + 1; j < N; j++) {
      if (samples[i] > samples[j]) {
        int t = samples[i]; samples[i] = samples[j]; samples[j] = t;
      }
    }
  }

  long sum = 0;
  for (int i = 5; i < N - 5; i++) sum += samples[i];
  float avg = float(sum) / (N - 10);
  return avg * 3.3 / 4095.0;
}

float computePH(float voltage) {
  float slope, ph;
  if (voltage < (cal4_voltage + cal7_voltage) / 2.0) {
    slope = (7.0 - 4.0) / (cal7_voltage - cal4_voltage);
    ph = 7.0 - (cal7_voltage - voltage) * slope;
  } else if (voltage < (cal7_voltage + cal9_voltage) / 2.0) {
    slope = (9.0 - 7.0) / (cal9_voltage - cal7_voltage);
    ph = 7.0 - (cal7_voltage - voltage) * slope;
  } else {
    slope = (9.0 - 7.0) / (cal9_voltage - cal7_voltage);
    ph = 7.0 - (cal7_voltage - voltage) * slope;
  }
  return ph;
}

// =========================
// 🖨️ Print Serial Output
// =========================
void printSensorData() {
  float phVoltage = readVoltage();
  float ph = computePH(phVoltage);
  lastVoltage = phVoltage;
  lastPH = ph;
  Serial.print("ADC Value: ");
  Serial.print(analogRead(tdsPin));
  Serial.print(" | TDS Voltage: ");
  Serial.print(voltage, 3);
  Serial.print(" V | TDS: ");
  Serial.print(tdsValue, 1);
  Serial.print(" ppm | pH Voltage: ");
  Serial.print(phVoltage, 3);
  Serial.print(" V | pH: ");
  Serial.println(ph, 2);
}

// =========================
// 🌐 Web Handlers
// =========================
void handleRoot() {
  String html = R"(
    <!DOCTYPE html>
    <html>
    <head>
      <title>ESP32 TDS & pH Monitor</title>
      <meta name='viewport' content='width=device-width, initial-scale=1'>
      <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; background: #f0f0f0; }
        .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .sensor-box { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        .value { font-size: 48px; color: #2196F3; font-weight: bold; margin: 20px 0; }
        .unit { font-size: 24px; color: #666; }
        .voltage { font-size: 18px; color: #888; margin-top: 10px; }
        .info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px; }
      </style>
      <script>
        function updateData() {
          fetch('/data')
          .then(response => response.json())
          .then(data => {
            document.getElementById('tds').innerText = data.tds.toFixed(1);
            document.getElementById('tds_voltage').innerText = data.voltage.toFixed(3);
            document.getElementById('ph').innerText = data.ph.toFixed(2);
            document.getElementById('ph_voltage').innerText = data.ph_voltage.toFixed(3);
          });
        }
        setInterval(updateData, 1000);
        updateData();
      </script>
    </head>
    <body>
      <div class='container'>
        <h1>TDS & pH Sensor Monitor</h1>
        <div class='sensor-box'>
          <h2>TDS Sensor</h2>
          <div class='value'><span id='tds'>0</span></div>
          <div class='unit'>PPM (parts per million)</div>
          <div class='voltage'>Voltage: <span id='tds_voltage'>0</span>V</div>
          <div class='info'>
            <p><strong>TDS Ranges:</strong></p>
            <p>0-50: Excellent | 50-100: Good<br>100-200: Fair | 200+: Poor</p>
          </div>
        </div>
        <div class='sensor-box'>
          <h2>pH Sensor</h2>
          <div class='value'><span id='ph'>0</span></div>
          <div class='unit'>pH Level</div>
          <div class='voltage'>Voltage: <span id='ph_voltage'>0</span>V</div>
        </div>
      </div>
    </body>
    </html>
  )";
  server.send(200, "text/html", html);
}

void handleTDS() {
  String response = "TDS: " + String(tdsValue, 1) + " ppm";
  server.send(200, "text/plain", response);
}

void handleData() {
  String json = "{\"tds\":" + String(tdsValue, 2) + ",\"voltage\":" + String(voltage, 3) + ",\"ph\":" +
                String(lastPH, 2) + ",\"ph_voltage\":" + String(lastVoltage, 3) + "}";
  server.send(200, "application/json", json);
}

// =========================
// 🧩 MQTT Helper Functions
// =========================

// Connect to MQTT broker
void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("ESP32_TDS_pH")) {
      Serial.println("Connected to MQTT!");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      delay(2000);
    }
  }
}

// Publish sensor data to MQTT topic
void publishSensorData() {
  if (!client.connected()) {
    reconnect();
  }

  String payload = "{\"tds\":" + String(tdsValue, 2) + ", \"ph\":" + String(lastPH, 2) + "}";
  client.publish("sensor/tds_ph", payload.c_str());
  client.loop();
}
