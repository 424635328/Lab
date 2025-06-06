# 确保 main 和它依赖的函数都定义在 if 块之前
def function_defined_later():
    print("Hello from function_defined_later!")

def another_helper_for_main():
    print("Another helper for main.")

def main():
    print("Inside main")
    function_defined_later()
    another_helper_for_main()
    print("Main finished")

# 'if __name__ == "__main__":' 不在最后
if __name__ == "__main__":
    print("Script execution started via __main__ guard.")
    main() # main() 被调用，此时它所需的所有函数都已定义
    print("Script execution via __main__ guard finished.")

# 在此之后定义的函数，main() 是无法调用的。
# 但是，在 'if' 块之后执行的其他顶级代码可以调用它们。
def some_other_function_not_called_by_main():
    print("This function is defined after the main guard and not called by main.")

print("Some top-level code running after the __main__ guard.")
if True: # 只是为了演示
    some_other_function_not_called_by_main()

print("Script end.")