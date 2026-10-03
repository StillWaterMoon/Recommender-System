// Build beside ItemCF.h and ItemCF.cpp:
// g++ -std=c++17 -O2 -pthread main.cpp ItemCF.cpp -o itemcf_test
// Contract: highest-weight history first; previously seen items are allowed.
// Input edges are unique, weights positive, IDs and thread counts valid.
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "ItemCF.h"

namespace {
int passed = 0;
int failed = 0;

void check(bool ok, const std::string& name) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << '\n';
    ok ? ++passed : ++failed;
}

void printIds(const std::vector<int>& ids) {
    std::cout << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << ']';
}

void expectIds(const std::string& name, const std::vector<int>& actual,
               const std::vector<int>& expected) {
    check(actual == expected, name);
    if (actual != expected) {
        std::cout << "       expected: ";
        printIds(expected);
        std::cout << "  actual: ";
        printIds(actual);
        std::cout << '\n';
    }
}

bool near(float actual, double expected) {
    return std::isfinite(actual) && std::abs(actual - expected) < 1e-5;
}

std::vector<int> neighborIds(const ItemCF& cf, int item) {
    std::vector<int> ids;
    for (const auto& entry : cf.top_sim[item]) ids.push_back(entry.second);
    return ids;
}

void expectSimilarity(const ItemCF& cf, int a, int b, double expected) {
    const auto& row = cf.top_sim[a];
    const auto it = std::find_if(row.begin(), row.end(),
        [b](const auto& entry) { return entry.second == b; });
    const bool ok = it != row.end() && near(it->first, expected);
    check(ok, "similarity " + std::to_string(a) + " -> " + std::to_string(b));
    if (!ok) {
        std::cout << "       expected: " << expected << "  actual: ";
        if (it == row.end()) std::cout << "missing";
        else std::cout << it->first;
        std::cout << '\n';
    }
}

void weightedTests() {
    std::cout << "\n=== Weighted data and history selection ===\n";
    // User 0: item 1 (weight 5) > item 2 (weight 3) > item 0 (weight 1).
    // Each history item has a distinct strongest neighbor:
    // 0 -> 3, 1 -> 4, 2 -> 5, all with similarity 1/sqrt(2).
    // User 4 has no history; item 6 has no interactions.
    std::vector<std::tuple<int, int, float>> edges = {
        {0, 0, 1.0f}, {0, 1, 5.0f}, {0, 2, 3.0f},
        {1, 0, 1.0f}, {1, 3, 2.0f},
        {2, 1, 5.0f}, {2, 4, 2.0f},
        {3, 2, 3.0f}, {3, 5, 2.0f}
    };
    ItemCF cf(5, 7, 3, edges, 4);
    const double s = 1.0 / std::sqrt(2.0);
    // No truncation here: every item has at most three neighbors.
    // Compare weighted similarities by item ID, avoiding float tie assumptions.
    for (const auto& pair : std::vector<std::pair<int, int>>{
             {0, 1}, {0, 2}, {1, 2}}) {
        expectSimilarity(cf, pair.first, pair.second, 0.5);
        expectSimilarity(cf, pair.second, pair.first, 0.5);
    }
    for (int item = 0; item < 3; ++item) {
        expectSimilarity(cf, item, item + 3, s);
        expectSimilarity(cf, item + 3, item, s);
    }
    bool structureOk = cf.top_sim.size() == 7;
    for (int item = 0; structureOk && item < 7; ++item) {
        const std::size_t expectedSize = item < 3 ? 3 : (item < 6 ? 1 : 0);
        structureOk = cf.top_sim[item].size() == expectedSize;
        for (const auto& entry : cf.top_sim[item]) {
            structureOk = structureOk && entry.second != item;
        }
    }
    check(structureOk, "neighbor counts, no self-neighbors, isolated item");

    expectIds("highest-weight one history item", cf.query(0, 1, 1, 10), {4});
    expectIds("highest-weight two history items", cf.query(0, 2, 1, 10), {4, 5});
    expectIds("all history, weighted ranking", cf.query(0, 3, 1, 10), {4, 5, 3});
    expectIds("return-count limit", cf.query(0, 3, 1, 2), {4, 5});
    expectIds("query neighbor limit", cf.query(0, 1, 2, 10), {4, 2});
    expectIds("oversized history count", cf.query(0, 100, 1, 10), {4, 5, 3});
    expectIds("empty user", cf.query(4, 3, 3, 10), {});
    expectIds("zero history count", cf.query(0, 0, 3, 10), {});
    expectIds("zero query neighbor count", cf.query(0, 3, 0, 10), {});
    expectIds("zero result count", cf.query(0, 3, 3, 0), {});

    ItemCF single(5, 7, 3, edges, 1);
    bool same = single.top_sim.size() == cf.top_sim.size();
    for (std::size_t i = 0; same && i < cf.top_sim.size(); ++i) {
        same = single.top_sim[i].size() == cf.top_sim[i].size();
        for (std::size_t j = 0; same && j < cf.top_sim[i].size(); ++j) {
            same = single.top_sim[i][j].second == cf.top_sim[i][j].second
                && near(single.top_sim[i][j].first, cf.top_sim[i][j].first);
        }
    }
    check(same, "one thread vs four threads: similarities");
    for (int user = 0; user < 5; ++user) {
        expectIds("one thread vs four threads: user " + std::to_string(user),
                  cf.query(user, 3, 3, 10), single.query(user, 3, 3, 10));
    }
}

void tieAndTruncationTests() {
    std::cout << "\n=== Exact ties, Top-K, and candidate accumulation ===\n";
    // One user, three items: every pair has exactly representable similarity 1.
    // This test intentionally permits recommendations of previously seen items.
    std::vector<std::tuple<int, int, float>> edges = {
        {0, 0, 1.0f}, {0, 1, 1.0f}, {0, 2, 1.0f}
    };
    ItemCF topOne(1, 3, 1, edges, 2);
    expectIds("Top-1 tie: item 0", neighborIds(topOne, 0), {2});
    expectIds("Top-1 tie: item 1", neighborIds(topOne, 1), {2});
    expectIds("Top-1 tie: item 2", neighborIds(topOne, 2), {1});
    expectIds("candidate contributions accumulate", topOne.query(0, 3, 1, 5), {2, 1});
    expectIds("query K above stored K", topOne.query(0, 3, 100, 5), {2, 1});

    ItemCF topTwo(1, 3, 2, edges, 2);
    expectIds("Top-2 tie: item 0", neighborIds(topTwo, 0), {2, 1});
    expectIds("Top-2 tie: item 1", neighborIds(topTwo, 1), {2, 0});
    expectIds("Top-2 tie: item 2", neighborIds(topTwo, 2), {1, 0});
    expectIds("equal scores: larger ID first", topTwo.query(0, 3, 2, 5), {2, 1, 0});
    expectIds("result cutoff on equal scores", topTwo.query(0, 3, 2, 2), {2, 1});

    ItemCF topZero(1, 3, 0, edges, 2);
    check(std::all_of(topZero.top_sim.begin(), topZero.top_sim.end(),
                     [](const auto& row) { return row.empty(); }),
          "constructor K=0 stores no neighbors");
    expectIds("constructor K=0 produces no candidates", topZero.query(0, 3, 3, 5), {});
}

void emptyDataTests() {
    std::cout << "\n=== Empty interaction data ===\n";
    std::vector<std::tuple<int, int, float>> edges;
    ItemCF cf(2, 3, 2, edges, 2);
    check(cf.top_sim.size() == 3 &&
          std::all_of(cf.top_sim.begin(), cf.top_sim.end(),
                      [](const auto& row) { return row.empty(); }),
          "empty data has no similar items");
    expectIds("empty data query", cf.query(0, 10, 10, 10), {});
}
} // namespace

int main() {
    weightedTests();
    tieAndTruncationTests();
    emptyDataTests();
    std::cout << "\nTOTAL: " << passed << " passed, " << failed << " failed\n";
    if (failed) {
        std::cout << "If history-selection checks fail, sort e[u] by descending weight.\n";
    }
    return failed == 0 ? 0 : 1;
}