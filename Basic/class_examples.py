
class Fun(int):

    def add(a, b):  # public method
        return a + b

    def _sub(a, b):  # protected method
        return a - b

    def __mult(a, b):  # private method
        return a * b

    def __special_dev__(a, b):
        return a/b

    @staticmethod
    def __stat(): ...

    @classmethod
    def __clas(): ...


class ExtraFun(Fun):

    def __mult(a, b):
        return super().__mult(b)
