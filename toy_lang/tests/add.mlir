module {
  func.func @main() -> !toy.int {
    %0 = toy.const 1 : !toy.int
    %1 = toy.const 2 : !toy.int
    %2 = toy.const 3 : !toy.int
    %3 = toy.add %2, %1 : !toy.int
    %4 = toy.add %3, %0 : !toy.int
    return %4 : !toy.int
  }
}