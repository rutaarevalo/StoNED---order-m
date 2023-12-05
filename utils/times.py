from datetime import datetime


def now_date():
    now = datetime.now()
    current_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Current Time =", current_time)
