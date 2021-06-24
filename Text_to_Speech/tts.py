
import pyttsx3


def speak(desc, objects, position):
    template = ' There is a {} to your {}'
    engine = pyttsx3.init()
    engine.say(desc)

    for i in range(len(objects)):
        # speak location of all the objects around
        engine.say(template.format(objects[i], position[i]))

    engine.runAndWait()
