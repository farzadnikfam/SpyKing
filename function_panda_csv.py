"""Panda Dataframe and CSV Creation"""

import glob
import os
import pandas as pd
from function_telegram_bot import telegram_bot_file
from function_telegram_bot import telegram_bot_image


def panda_csv(panda_log, logpath, file_name="coccobello", print_dataframe=True, send_dataframe=True, print_dataimage=False, send_dataimage=True, validation_flag=1):
    "create a CSV file from pandas dataframe"

    file_path = logpath + "/%s" %(file_name) #creates a path to record results
    panda_log_float = panda_log.astype(float)
    panda_log.to_csv(file_path + ".csv", index=None)

    # printing all data in figures
    panda_log_lr = panda_log_float.plot(y='lr', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False)
    panda_log_acc_train = panda_log_float.plot(y='acc_train', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False)
    if validation_flag: panda_log_acc_val = panda_log_float.interpolate(method='linear').plot(y='acc_val', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False) #interpolation to pass the NaN problem if the validation frequency is not 1
    panda_log_loss_train = panda_log_float.plot(y='loss_train', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False)
    if validation_flag: panda_log_loss_val = panda_log_float.interpolate(method='linear').plot(y='loss_val', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False) #interpolation to pass the NaN problem if the validation frequency is not 1

    # saving all figures
    panda_log_lr.get_figure().savefig(file_path + "_lr.png")
    panda_log_acc_train.get_figure().savefig(file_path + "_acc_train.png")
    if validation_flag: panda_log_acc_val.get_figure().savefig(file_path + "_acc_val.png")
    panda_log_loss_train.get_figure().savefig(file_path + "_loss_train.png")
    if validation_flag: panda_log_loss_val.get_figure().savefig(file_path + "_loss_val.png")

    # shows all figures
    if print_dataimage == True:
        panda_log_lr.figure.show()
        panda_log_acc_train.figure.show()
        if validation_flag: panda_log_acc_val.figure.show()
        panda_log_loss_train.figure.show()
        if validation_flag: panda_log_loss_val.figure.show()

    # sends all figures to telegram bot
    if send_dataimage == True:
        telegram_bot_image(file_path + "_lr.png", "learning rate")
        telegram_bot_image(file_path + "_acc_train.png", "accuracy train")
        if validation_flag: telegram_bot_image(file_path + "_acc_val.png", "accuracy validation")
        telegram_bot_image(file_path + "_loss_train.png", "loss train")
        if validation_flag: telegram_bot_image(file_path + "_loss_val.png", "loss validation")

    if print_dataframe == True: #prints the dataframe
        print(panda_log)
    if send_dataframe == True: #sends dataframe to telegram
        telegram_bot_file(file_path+ ".csv")

    return panda_log.to_csv(file_path + ".csv", index=None)