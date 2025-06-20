"""Telegram Bot"""

import requests
import re


# before using this function putting True you have to compile the "telegram_bot.txt" file with your Chat ID and the Bot Token
# put False if you don't love bots or don't want spam on your phone, but you need to know that I'm not your friend anymore!!!
i_love_bot = False #True or False

# reads the "telegram_bot.txt" file if it exists
try:
   file = open("telegram_bot.txt", "r") #reading mode
   for line in file:
      if line.find('bot_token:') != -1: bot_token = re.search("'(.+?)'", line).group(1) #records the Bot Token
      if line.find('bot_chatID:') != -1: bot_chatID = re.search("'(.+?)'", line).group(1) #records the Chat ID
   file.close()
except:
   i_love_bot = False

# for sending text messages
def telegram_bot_text(bot_message, bold = False):
   "send notification via telegram bot"
   if bold == False:
      bot_message = bot_message.replace("#","%23") #to replace the hashtag with a format compatible with the telegram API
      bot_message = bot_message.replace("_","\_") #to replace the underscore with a format compatible with the telegram API
   else:
      bot_message = "*" + bot_message + "*"
   if i_love_bot == True:
      try:
         send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
         requests.get(send_text)
      except:
         print("Message NOT sent, there are problems communicating with Telegram!")
   else:
      # nothing will happen
      pass

# for sending text messages and print on screen
def telegram_bot_text_print(bot_message, bold = False):
   "send notification via telegram bot"
   print(bot_message)
   if bold == False:
      bot_message = bot_message.replace("#","%23") #to replace the hashtag with a format compatible with the telegram API
      bot_message = bot_message.replace("_","\_") #to replace the underscore with a format compatible with the telegram API
      bot_message = bot_message.replace("[","\[") #to replace the square brackets with a format compatible with the telegram API
   else:
      bot_message = "*" + bot_message + "*"
   if i_love_bot == True:
      try:
         send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
         requests.get(send_text)
      except:
         print("Message NOT sent, there are problems communicating with Telegram!")
   else:
      # nothing will happen
      pass

# for sending files
def telegram_bot_file(bot_file):
   "send file via telegram bot"
   if i_love_bot == True:
      try:
         files = {'document': open(bot_file, 'rb')}
         send_text = 'https://api.telegram.org/bot' + bot_token + '/sendDocument?chat_id=' + bot_chatID
         requests.post(send_text, files=files)
      except:
         print("File NOT sent, there are problems communicating with Telegram!")
   else:
      # nothing will happen
      pass

# for sending images with captions
def telegram_bot_image(bot_image, bot_message=None):
   "send image via telegram bot"
   bot_image = r'%s' %(bot_image) #the "r" reads the string as it is without interpreting characters as "\n"
   bot_message = bot_message.replace("#","%23") #to replace the hashtag with a format compatible with the telegram API
   bot_message = bot_message.replace("_","\_") #to replace the underscore with a format compatible with the telegram API
   if i_love_bot == True:
      try:
         files = {'photo': open(bot_image, 'rb')}
         if bot_message == None:
            send_text = 'https://api.telegram.org/bot' + bot_token + '/sendPhoto?chat_id=' + bot_chatID
         else:
            send_text = 'https://api.telegram.org/bot' + bot_token + '/sendPhoto?chat_id=' + bot_chatID + '&parse_mode=Markdown&caption=' + bot_message
         requests.post(send_text, files=files)
      except:
         print("Image NOT sent, there are problems communicating with Telegram!")
   else:
      # nothing will happen
      pass