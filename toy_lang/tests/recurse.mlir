func.func @fib(%arg_0 : !toy.int) -> (!toy.int) {
  %x0 = toy.const 2 : !toy.int
  %x1 = toy.less %x0, %arg_0 : !toy.int
  %x2 = toy.if %x1 : !toy.int {
    %x3 = toy.const 1 : !toy.int
    toy.yield %x3
  } {
    %x4 = toy.const 1 : !toy.int
    %x5 = toy.neg %x4 : !toy.int
    %x6 = toy.add %x5, %arg_0 : !toy.int
    %x7 = func.call @fib (%x6) : (!toy.int) -> (!toy.int)
    %x8 = toy.const 2 : !toy.int
    %x9 = toy.neg %x8 : !toy.int
    %x10 = toy.add %arg_0, %x9 : !toy.int
    %x11 = func.call @fib (%x10) : (!toy.int) -> (!toy.int)
    %x12 = toy.add %x11, %x7 : !toy.int
    toy.yield %x12
  }
  return %x2 : !toy.int
}
func.func @main() -> (!toy.int) {
  %x0 = toy.const 10 : !toy.int
  %x1 = func.call @fib (%x0) : (!toy.int) -> (!toy.int)
  return %x1 : !toy.int
}