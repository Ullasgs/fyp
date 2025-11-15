const int RELAY_PIN = A0;  // Relay control pin
bool pumpRunning = false;
bool startCommandReceived = false;
unsigned long startDelayTime = 0;
unsigned long pumpStartTime = 0;

void setup() {
  Serial.begin(9600);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // Pump OFF at start (for Active LOW relay)
  Serial.println("System ready. Type START to begin.");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.equalsIgnoreCase("START") && !pumpRunning && !startCommandReceived) {
      startCommandReceived = true;
      startDelayTime = millis();
    }
  }

  // After 5-second delay, start pump
  if (startCommandReceived && !pumpRunning && millis() - startDelayTime >= 1000) {
    pumpRunning = true;
    startCommandReceived = false;
    pumpStartTime = millis();
    digitalWrite(RELAY_PIN, LOW); // Turn pump ON (Active LOW)
    Serial.println("✅ Pump ON");
  }

  // Stop pump after 5 seconds
  if (pumpRunning && millis() - pumpStartTime >= 3000) {
    pumpRunning = false;
    digitalWrite(RELAY_PIN, HIGH); // Turn pump OFF (Active LOW)
    Serial.println("🛑 Pump OFF (Calibrating current TDS)");
  }
}