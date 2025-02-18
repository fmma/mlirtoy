func.func @foo(%arg_0 : i32, %arg_1 : i32) -> (i32, i32) {
  return %arg_0, %arg_1 : i32, i32
}
func.func @main() -> (i32, i32) {
  %x0 = toy.constant 1 : i32
  %x1 = toy.constant 2 : i32
  %x2, %x3 = func.call @foo (%x1, %x0) : (i32, i32) -> (i32, i32)
  return %x2, %x3 : i32, i32
}