import alphaCalculator as Alpha


def run_alpha_calculator(alpha_index):
    alpha_method_name = f'alpha_{alpha_index:03}'
    if hasattr(alpha_calculator, alpha_method_name):
        alpha_method = getattr(alpha_calculator, alpha_method_name)
        result = alpha_method()  # 调用相应的计算方法
        print(f"因子%s的值为：{result}" % alpha_method_name)
    else:
        print("请输入有效的 alpha 因子编号！")


if __name__ == '__main__':
    alpha_calculator = Alpha.GTJA_191()
    print("你可以通过输入set调整日期和股票！")
    while True:
        try:
            my_string = input("如要测试请输入 alpha 因子三位数字编号（例如：089）：")
            if 's' in my_string:
                start_date = input("请输入起始日期（例如：2013-01-04）：")
                end_date = input("请输入终止日期（例如：2024-03-04）：")
                start_stock = input("请输入起始日期（例如：000001.SZ）：")
                ebd_stock = input("请输入终止日期（例如：689009.SH）：")
                alpha_calculator.set_date_and_stock(start_date=start_date, end_date=end_date,
                                                  start_stock=start_stock, end_stock=ebd_stock)
            else:
                run_alpha_calculator(my_string)
        except KeyboardInterrupt as q:
            print("\n测试结束。")
            break
        except Exception as e:
            print(f"发生错误：{e}")
