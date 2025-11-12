/*
  esp32.ino
  Simulated TDS (700-900 ppm), smooth transitions, publishes to MQTT and serves /data over HTTP.
  Includes simple relay control endpoints /relay/on and /relay/off.
*/

#include <WiFi.h>
#include <WebServer.h>
#include <PubSubClient.h>

// ---------- CONFIG ----------
const char* ssid = "Phone";
const char* password = "ullasgss";

const char* mqtt_server = "172.20.10.10"; // your broker
const int mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);
WebServer server(80);

// ADC / sensors (we're simulating TDS)
const int tdsPin = 34;
#define PH_PIN 35

// pH calibration values (kept from your original)
float cal4_voltage = 1.7461;
float cal7_voltage = 1.8797;
float cal9_voltage = 2.039;

float temperature = 25.0;
float K_VALUE = 0.60;

float voltage = 0.0;
float tdsValue = 0.0;
float lastVoltage = 0.0;
float lastPH = 0.0;

// SIMULATION CONTROL
const bool SIMULATE_TDS = true;   // true => uses simulated TDS instead of ADC
const float SIM_MIN = 700.0;      // normal lower bound
const float SIM_MAX = 900.0;      // normal upper bound
const unsigned long SIM_STEP_MS = 500; // how often we compute new small change
float simulatedTarget = 800.0;    // initial internal smoothing target

// RELAY
const int RELAY_PIN = 2; // change to the pin you use; avoid A0 on many boards for digital control

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  delay(2000);

  analogReadResolution(12);
  analogSetPinAttenuation(tdsPin, ADC_11db);
  analogSetPinAttenuation(PH_PIN, ADC_11db);
  pinMode(tdsPin, INPUT);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // default off

  randomSeed(micros());

  connectToWiFi();

  client.setServer(mqtt_server, mqtt_port);

  server.on("/", handleRoot);
  server.on("/tds", handleTDS);
  server.on("/data", handleData);
  server.on("/relay/on", [](){ digitalWrite(RELAY_PIN, HIGH); server.send(200,"text/plain","OK"); });
  server.on("/relay/off", [](){ digitalWrite(RELAY_PIN, LOW); server.send(200,"text/plain","OK"); });

  server.begin();

  Serial.println("HTTP server started");
  Serial.println("=== ESP32 TDS & pH Simulator (700-900 ppm) ===");

  if (SIMULATE_TDS) {
    // initialize simulatedTarget near midpoint
    simulatedTarget = (SIM_MIN + SIM_MAX) * 0.5;
    tdsValue = simulatedTarget;
  }
}

// ---------- LOOP ----------
void loop() {
  server.handleClient();
  static unsigned long lastPub = 0;
  static unsigned long lastSim = 0;

  // smooth simulation tick
  if (SIMULATE_TDS && (millis() - lastSim >= SIM_STEP_MS)) {
    lastSim = millis();
    simulateTDSStep();
  }

  // publish every 1s
  if (millis() - lastPub >= 1000) {
    lastPub = millis();

    if (!SIMULATE_TDS) readTDSSensor(); // real ADC path if turned off
    printSensorData();
    publishSensorData();
  }

  simulateTemperature();
}

// ---------- WiFi ----------
void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(250);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi");
  }
}

// ---------- TDS ADC (kept, not used when SIMULATE_TDS true) ----------
void readTDSSensor() {
  const int samples = 30;
  long adcValue = 0;
  for (int i = 0; i < samples; i++) {
    adcValue += analogRead(tdsPin);
    delay(3);
  }
  adcValue /= samples;
  const float VREF = 3.3;
  const int ADC_RES = 4095; // ESP32 ADC read range
  voltage = (adcValue / (float)ADC_RES) * VREF;
  float comp = 1.0 + 0.02 * (temperature - 25.0);
  float compV = voltage / comp;
  if (compV < 0.1)
    tdsValue = (compV / VREF) * 500 * K_VALUE;
  else
    tdsValue = (133.42 * compV * compV * compV
                - 255.86 * compV * compV
                + 857.39 * compV) * K_VALUE;
  if (tdsValue < 0) tdsValue = 0;
}

// ---------- pH ----------
float readVoltage() {
  const int N = 30;
  int samples[N];
  for (int i = 0; i < N; i++) {
    samples[i] = analogRead(PH_PIN);
    delay(3);
  }
  for (int i = 0; i < N - 1; i++)
    for (int j = i + 1; j < N; j++)
      if (samples[i] > samples[j]) {
        int t = samples[i];
        samples[i] = samples[j];
        samples[j] = t;
      }
  long sum = 0;
  for (int i = 5; i < N - 5; i++) sum += samples[i];
  float avg = float(sum) / (N - 10);
  return avg * 3.3 / 4095.0;
}

float computePH(float voltage_) {
  float slope, ph;
  if (voltage_ < (cal4_voltage + cal7_voltage) / 2.0) {
    slope = (7.0 - 4.0) / (cal7_voltage - cal4_voltage);
    ph = 7.0 - (cal7_voltage - voltage_) * slope;
  } else if (voltage_ < (cal7_voltage + cal9_voltage) / 2.0) {
    slope = (9.0 - 7.0) / (cal9_voltage - cal7_voltage);
    ph = 7.0 - (cal7_voltage - voltage_) * slope;
  } else {
    slope = (9.0 - 7.0) / (cal9_voltage - cal7_voltage);
    ph = 7.0 - (cal7_voltage - voltage_) * slope;
  }
  return ph;
}

// ---------- Print ----------
void printSensorData() {
  float phVoltage = readVoltage();
  float ph = computePH(phVoltage);
  lastVoltage = phVoltage;
  lastPH = ph;

  Serial.print("ADC: "); Serial.print(analogRead(tdsPin));
  Serial.print(" | TDS: "); Serial.print(tdsValue, 2);
  Serial.print(" ppm | pH: "); Serial.print(ph, 2);
  Serial.print(" | Temp: "); Serial.println(temperature, 2);
}

// ---------- HTTP handlers ----------
void handleRoot() {
  String html = "<html><body><h3>ESP32 TDS Simulator</h3><p>Visit /data</p></body></html>";
  server.send(200, "text/html", html);
}

void handleTDS() {
  String response = "TDS: " + String(tdsValue, 2) + " ppm";
  server.send(200, "text/plain", response);
}

void handleData() {
  String json = "{\"tds\":" + String(tdsValue, 2) + ",\"ph\":" + String(lastPH, 2) + ",\"temperature\":" + String(temperature, 2) + "}";
  server.send(200, "application/json", json);
}

// ---------- MQTT ----------
void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("ESP32_TDS_pH")) {
      Serial.println("Connected to MQTT!");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      Serial.println();
      delay(2000);
    }
  }
}

void publishSensorData() {
  if (!client.connected()) reconnect();
  String payload = "{\"tds\":" + String(tdsValue, 2) + ", \"ph\":" + String(lastPH, 2) + ", \"temperature\":" + String(temperature, 2) + "}";
  client.publish("sensor/tds_ph", payload.c_str());
  client.loop();
}

// ---------- Simulate temperature ----------
void simulateTemperature() {
  static float currentTemp = 25.0;
  static unsigned long lastTempUpdate = 0;
  if (millis() - lastTempUpdate >= 1500) {
    lastTempUpdate = millis();
    float variation = ((float)random(-15, 16)) / 100.0;
    currentTemp += variation;
    if (currentTemp < 24.0) currentTemp = 24.0;
    if (currentTemp > 28.0) currentTemp = 28.0;
    temperature = currentTemp;
  }
}

// ---------- Simulated TDS step (smooth, small jitter, stabilizes ~10s) ----------
void simulateTDSStep() {
  if (random(0, 100) < 30) {
    float nudge = ((float)random(-50, 51));
    simulatedTarget += nudge * 0.02;
  }
  if (simulatedTarget < SIM_MIN) simulatedTarget = SIM_MIN;
  if (simulatedTarget > SIM_MAX) simulatedTarget = SIM_MAX;

  float alpha = 0.15;
  float noise = ((float)random(-100, 101)) / 100.0;
  tdsValue = tdsValue * (1.0 - alpha) + (simulatedTarget + noise) * alpha;

  if (tdsValue < SIM_MIN - 5.0) tdsValue = SIM_MIN;
  if (tdsValue > SIM_MAX + 5.0) tdsValue = SIM_MAX;
}
