import RPi.GPIO as GPIO
import time

# Motor trai tren board TB6612FNG
AIN1 = 20
AIN2 = 21
PWMA = 16
STBY = 12

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(STBY, GPIO.OUT)

pwm = GPIO.PWM(PWMA, 1000)
pwm.start(0)

def enable_driver():
    GPIO.output(STBY, GPIO.HIGH)

def forward(speed=100):
    enable_driver()
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)
    pwm.ChangeDutyCycle(speed)

def backward(speed=100):
    enable_driver()
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.HIGH)
    pwm.ChangeDutyCycle(speed)

def stop():
    pwm.ChangeDutyCycle(0)
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.LOW)

try:
    print("Motor trai quay chieu 1 - 100%")
    forward(100)
    time.sleep(5)

    print("Dung")
    stop()
    time.sleep(2)

    print("Motor trai quay chieu 2 - 100%")
    backward(100)
    time.sleep(5)

    print("Dung")
    stop()

except KeyboardInterrupt:
    print("Dung bang Ctrl+C")

finally:
    stop()
    GPIO.output(STBY, GPIO.LOW)
    pwm.stop()
    GPIO.cleanup()
    print("Da tat motor va cleanup GPIO")