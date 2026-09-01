#include "batch_scheduler.hpp"
#include <cassert>

int main() {
  cbllm::BatchScheduler scheduler(2, 100);
  scheduler.enqueue({"low", 10, 1, 0});
  scheduler.enqueue({"high", 10, 9, 1});
  const auto batch = scheduler.schedule({});
  assert(batch.size() == 2);
  assert(batch.front().id == "high");
}

