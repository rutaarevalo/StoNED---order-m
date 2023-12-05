# import packages

from utils.dataset import load_data_electric
from utils.method_m import METHOD_M
from utils.times import now_date


def main():
    print('init')
    now_date()
    """load data"""
    data = load_data_electric()

    """setup model data"""
    process = METHOD_M(data)
    """execute stoNED"""
    try:
        process.execute()
    except Exception as error:
        print(error.__cause__)

    print("finish process")


if __name__ == '__main__':
    main()
