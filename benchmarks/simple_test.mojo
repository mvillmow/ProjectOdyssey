struct TestConfig:
    var value: Int

    fn __init__(
        inout self,
        val: Int = 10,
    ):
        self.value = val


fn main():
    var config = TestConfig(5)
    print("Value:", config.value)
