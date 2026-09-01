#include "batch_scheduler.hpp"
#include <algorithm>

namespace cbllm {
BatchScheduler::BatchScheduler(std::size_t max_sequences, std::size_t max_tokens)
    : max_sequences_(max_sequences), max_tokens_(max_tokens) {}

bool BatchScheduler::Compare::operator()(const Sequence& left, const Sequence& right) const {
  if (left.priority != right.priority) return left.priority < right.priority;
  return left.arrival_order > right.arrival_order;
}

void BatchScheduler::enqueue(Sequence sequence) { waiting_.push(std::move(sequence)); }

std::vector<Sequence> BatchScheduler::schedule(const std::vector<Sequence>& running) {
  auto batch = running;
  std::sort(batch.begin(), batch.end(), [](const auto& a, const auto& b) {
    return a.priority == b.priority ? a.arrival_order < b.arrival_order : a.priority > b.priority;
  });
  if (batch.size() > max_sequences_) batch.resize(max_sequences_);
  std::size_t tokens = 0;
  for (const auto& sequence : batch) tokens += sequence.tokens;
  while (!waiting_.empty() && batch.size() < max_sequences_) {
    auto next = waiting_.top();
    if (tokens + next.tokens > max_tokens_) break;
    waiting_.pop();
    tokens += next.tokens;
    batch.push_back(std::move(next));
  }
  return batch;
}
}  // namespace cbllm

