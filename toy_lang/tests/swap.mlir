func.func @foo(%arg_0 : !toy.int, %arg_1 : !toy.int) -> (!toy.int, !toy.int) {
  return %arg_0, %arg_1 : !toy.int, !toy.int
}
func.func @main() -> (!toy.int, !toy.int) {
  %x0 = arith.constant 1 : i32
  %x1 = toy.from_i32 %x0 : i32 -> !toy.int
  %x2 = arith.constant 2 : i32
  %x3 = toy.from_i32 %x2 : i32 -> !toy.int
  %x4, %x5 = func.call @foo (%x3, %x1) : (!toy.int, !toy.int) -> (!toy.int, !toy.int)
  return %x4, %x5 : !toy.int, !toy.int
}