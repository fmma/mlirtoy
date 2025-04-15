func.func @choice(%arg_0 : !toy.int, %arg_1 : !toy.int) -> (!toy.int) {
  %x0 = toy.if %arg_0 : !toy.int {
  %x1 = toy.const 10 : !toy.int
  %x2 = toy.mul %x1, %arg_1 : !toy.int
  toy.yield %x2
  } {
  %x3 = toy.neg %arg_1 : !toy.int
  toy.yield %x3
  }
  return %x0 : !toy.int
}
func.func @nested_choice_1(%arg_0 : !toy.int, %arg_1 : !toy.int) -> (!toy.int) {
  %x0 = toy.if %arg_0 : !toy.int {
  %x1 = toy.const 1 : !toy.int
  toy.yield %x1
  } {
  %x2 = toy.if %arg_1 : !toy.int {
  %x3 = toy.const 2 : !toy.int
  toy.yield %x3
  } {
  %x4 = toy.const 3 : !toy.int
  toy.yield %x4
  }
  toy.yield %x2
  }
  return %x0 : !toy.int
}
func.func @nested_choice_2(%arg_0 : !toy.int, %arg_1 : !toy.int) -> (!toy.int) {
  %x0 = toy.if %arg_0 : !toy.int {
  %x1 = toy.if %arg_1 : !toy.int {
  %x2 = toy.const 1 : !toy.int
  toy.yield %x2
  } {
  %x3 = toy.const 2 : !toy.int
  toy.yield %x3
  }
  toy.yield %x1
  } {
  %x4 = toy.const 3 : !toy.int
  toy.yield %x4
  }
  return %x0 : !toy.int
}
func.func @main() -> (!toy.int) {
  %x0 = toy.const 100 : !toy.int
  %x1 = toy.const 1 : !toy.int
  %x2 = func.call @choice (%x1, %x0) : (!toy.int, !toy.int) -> (!toy.int)
  %x3 = func.call @nested_choice_1 (%x2, %x2) : (!toy.int, !toy.int) -> (!toy.int)
  %x4 = func.call @nested_choice_2 (%x3, %x3) : (!toy.int, !toy.int) -> (!toy.int)
  return %x4 : !toy.int
}