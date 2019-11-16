# Import the required module for text
# to speech conversion
from gtts import gTTS

# This module is imported so that we can
# play the converted audio
import os

# The text that you want to convert to audio
mytext = """Bilindiği üzere karapara aklama suçu kavramı mevzuatımıza ilk defa 4208 sayılı Karaparanın Aklanmasının Önlenmesi Hakkında Kanun ile gelmiştir. Yükümlülüklerin konu olarak yer almadığı 4208 sayılı Kanunda; karapara aklamanın öncül suçları belirlenmiş, karapara ve karapara aklama suçu tanımlanmış ve aklama suçunun cezası tespit edilmiştir. 
"""

# Language in which you want to convert
language = 'tr'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in a mp3 file named
# welcome
myobj.save("welcome.mp3")

# Playing the converted file
# os.system("mp3 welcome.mp3")