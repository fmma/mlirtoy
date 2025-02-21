func.func @main() -> (!toy.int) {
  %x0 = toy.const 1 : !toy.int
  %x1 = toy.const 2 : !toy.int
  %x2 = toy.const 3 : !toy.int
  %x3 = toy.add %x2, %x1 : !toy.int
  %x4 = toy.add %x3, %x0 : !toy.int
  return %x4 : !toy.int
}