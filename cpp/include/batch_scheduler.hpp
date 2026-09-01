#pragma once
#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

namespace cbllm {
struct Sequence {
  std::string id;
  std::size_t tokens;
  std::uint32_t priority;
  std::uint64_t arrival_order;
};

class BatchScheduler {
 public:
  BatchScheduler(std::size_t max_sequences, std::size_t max_tokens);
  void enqueue(Sequence sequence);
  std::vector<Sequence> schedule(const std::vector<Sequence>& running);
  [[nodiscard]] std::size_t waiting() const noexcept { return waiting_.size(); }

 private:
  struct Compare {
    bool operator()(const Sequence& left, const Sequence& right) const;
  };
  std::size_t max_sequences_;
  std::size_t max_tokens_;
  std::priority_queue<Sequence, std::vector<Sequence>, Compare> waiting_;
};
}  // namespace cbllm

