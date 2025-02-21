// RUN: toy-opt %s > %t
// RUN FileCheck %s < %t

module {
  // CHECK-LABEL: test_type_syntax
  func.func @test_type_syntax(%arg0: !toy.int) -> !toy.int {
    // CHECK: toy.int
    return %arg0 : !toy.int
  }

  // CHECK-LABEL: test_add_syntax
  func.func @test_add_syntax(%arg0: !toy.int, %arg1: !toy.int) -> !toy.int {
    %0 = toy.const 1 : !toy.int
    %1 = toy.add %arg0, %0 : !toy.int
    %2 = toy.get : !toy.int
    toy.put %2
    return %1 : !toy.int
  }
}
