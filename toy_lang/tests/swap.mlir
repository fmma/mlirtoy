module {
  func.func @foo(%arg0: !toy.int, %arg1: !toy.int) -> (!toy.int, !toy.int) {
    return %arg0, %arg1 : !toy.int, !toy.int
  }
  func.func @main() -> (!toy.int, !toy.int) {
    %0 = toy.const 1 : !toy.int
    %1 = toy.const 2 : !toy.int
    %2:2 = call @foo(%1, %0) : (!toy.int, !toy.int) -> (!toy.int, !toy.int)
    return %2#0, %2#1 : !toy.int, !toy.int
  }
}