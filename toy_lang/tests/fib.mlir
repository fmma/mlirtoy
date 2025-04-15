module {
  func.func @fib(%arg0: !toy.int) -> !toy.int {
    %0 = toy.const 2 : !toy.int
    %1 = toy.less %0, %arg0 : !toy.int
    %2 = toy.if %1 : !toy.int {
      %3 = toy.const 1 : !toy.int
      toy.yield %3
    } {
      %3 = toy.const 1 : !toy.int
      %4 = toy.neg %3 : !toy.int
      %5 = toy.add %4, %arg0 : !toy.int
      %6 = func.call @fib(%5) : (!toy.int) -> !toy.int
      %7 = toy.const 2 : !toy.int
      %8 = toy.neg %7 : !toy.int
      %9 = toy.add %arg0, %8 : !toy.int
      %10 = func.call @fib(%9) : (!toy.int) -> !toy.int
      %11 = toy.add %10, %6 : !toy.int
      toy.yield %11
    }
    return %2 : !toy.int
  }
  func.func @main() -> !toy.int {
    %0 = toy.const 10 : !toy.int
    %1 = call @fib(%0) : (!toy.int) -> !toy.int
    return %1 : !toy.int
  }
}