from flask import Flask
from gpiozero import Button, LED

red = LED(17)
blue = LED(27)
green = LED(22)
led_list = [red, blue, green]
button = Button(26, bounce_time=0.05)
app = Flask(__name__)

for led in led_list:
    led.off()

#Add text to the homepage
@app.route("/")
def index():
    return "Deez Nuts"

#Tell the user if the button is pressed or not on the push-button page
@app.route("/push-button")
def check_push_button_state():
    if button.is_pressed:
        return "Button is pressed"
    return "Button is not pressed"

#Turn an LED on or off depending on which URL the user enters
@app.route("/<int:led_number>/<int:state>")
def switch_led(led_number, state):
    if led_number < 0 or led_number >= len(led_list):
        return "Invalid LED Number: Please enter a number between 0 and " + str(len(led_list) - 1)
    if state == 1:
        led_list[led_number].on()
    elif state == 0:
        led_list[led_number].off()
    else:
        return "Please enter 0 to turn light off or 1 to turn on"
    return "OK"


app.run(host="0.0.0.0")