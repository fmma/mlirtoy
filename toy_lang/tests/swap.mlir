func.func @foo(%arg_0 : !toy.int, %arg_1 : !toy.int) -> (!toy.int, !toy.int) {
  return %arg_0, %arg_1 : !toy.int, !toy.int
}
func.func @main() -> (!toy.int, !toy.int) {
  %x0 = toy.const 1 : !toy.int
  %x1 = toy.const 2 : !toy.int
  %x2, %x3 = func.call @foo (%x1, %x0) : (!toy.int, !toy.int) -> (!toy.int, !toy.int)
  return %x2, %x3 : !toy.int, !toy.int
}