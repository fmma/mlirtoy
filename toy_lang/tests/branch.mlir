module {
  func.func @choice(%arg0: !toy.int, %arg1: !toy.int) -> !toy.int {
    %0 = toy.if %arg0 : !toy.int {
      %1 = toy.const 10 : !toy.int
      %2 = toy.mul %1, %arg1 : !toy.int
      toy.yield %2
    } {
      %1 = toy.neg %arg1 : !toy.int
      toy.yield %1
    }
    return %0 : !toy.int
  }
  func.func @nested_choice_1(%arg0: !toy.int, %arg1: !toy.int) -> !toy.int {
    %0 = toy.if %arg0 : !toy.int {
      %1 = toy.const 1 : !toy.int
      toy.yield %1
    } {
      %1 = toy.if %arg1 : !toy.int {
        %2 = toy.const 2 : !toy.int
        toy.yield %2
      } {
        %2 = toy.const 3 : !toy.int
        toy.yield %2
      }
      toy.yield %1
    }
    return %0 : !toy.int
  }
  func.func @nested_choice_2(%arg0: !toy.int, %arg1: !toy.int) -> !toy.int {
    %0 = toy.if %arg0 : !toy.int {
      %1 = toy.if %arg1 : !toy.int {
        %2 = toy.const 1 : !toy.int
        toy.yield %2
      } {
        %2 = toy.const 2 : !toy.int
        toy.yield %2
      }
      toy.yield %1
    } {
      %1 = toy.const 3 : !toy.int
      toy.yield %1
    }
    return %0 : !toy.int
  }
  func.func @main() -> !toy.int {
    %0 = toy.const 100 : !toy.int
    %1 = toy.const 1 : !toy.int
    %2 = call @choice(%1, %0) : (!toy.int, !toy.int) -> !toy.int
    %3 = call @nested_choice_1(%2, %2) : (!toy.int, !toy.int) -> !toy.int
    %4 = call @nested_choice_2(%3, %3) : (!toy.int, !toy.int) -> !toy.int
    return %4 : !toy.int
  }
}