// RUN: toy-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @test_simple
func.func @test_simple() -> !toy.int {
  // CHECK: toy.const 6
  // CHECK-NEXT: return
  %p0 = toy.const 3 : !toy.int
  %2 = toy.add %p0, %p0 : !toy.int
  %3 = toy.mul %p0, %p0 : !toy.int
  %4 = toy.add %2, %3 : !toy.int
  return %2 : !toy.int
}