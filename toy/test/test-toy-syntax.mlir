// RUN: toy-opt %s > %t
// RUN FileCheck %s < %t

module {
  // CHECK-LABEL: test_type_syntax
  func.func @test_type_syntax(%arg0: !toy.int) -> !toy.int {
    // CHECK: toy.int
    return %arg0 : !toy.int
  }

  // CHECK-LABEL: test_add_syntax
  func.func @test_add_syntax(%arg0: !toy.int<10>, %arg1: !toy.int<10>) -> !toy.int<10> {
    // CHECK: toy.add
    %0 = toy.add %arg0, %arg1 : !toy.int<10>
    return %0 : !toy.int<10>
  }
}
