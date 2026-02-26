#include <Wire.h>
#include <Adafruit_BMP085.h>
#include <MPU6050.h>

Adafruit_BMP085 bmp;
MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); 

  if (!bmp.begin()) {
    Serial.println("BMP180 not found!");
    return;
  }

  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 not found!");
    return;
  }
}

void loop() {
 
  float bmpTemp = bmp.readTemperature();

  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  float mpuTemp = mpu.getTemperature() / 340.0 + 36.53;

  Serial.println("------");
  Serial.print("BMP180 Temp: ");
  Serial.print(bmpTemp);
  Serial.println(" C");

  Serial.print("MPU6050 Temp: ");
  Serial.print(mpuTemp);
  Serial.println(" C");

  Serial.print("Accel Y: ");
  Serial.println(ay);
  Serial.print("Accel Z: ");
  Serial.println(az);

  delay(1000);
}
