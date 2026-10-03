// Redis-backed ItemCF integration tests (C++17).
// Place beside your Redis version of ItemCF.h and ItemCF.cpp.
// Build: g++ -std=c++17 -O2 -pthread main.cpp ItemCF.cpp -lredis++ -lhiredis -o itemcf_test
// Run against a dedicated EMPTY test database:
// ./itemcf_test tcp://127.0.0.1:6379/15
// Do not use a database shared with another running application.
// No FLUSHDB/FLUSHALL is used. Only this fixture's named keys are removed.
// Preconditions: valid positive query counts, canonical integer item IDs,
// positive weights, and meta:n/meta:m always present during model construction.

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <sw/redis++/redis++.h>
#include "ItemCF.h"

namespace {
using Row = std::vector<std::pair<std::string, double>>;
using Edge = std::tuple<int, int, double>;
int passed = 0;
int failed = 0;

void check(bool ok, const std::string& name) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << '\n';
    if (ok) ++passed;
    else ++failed;
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

Row readNeighbors(sw::redis::Redis& redis, int item) {
    Row row;
    redis.zrevrange("top_sim:" + std::to_string(item), 0, -1,
                   std::back_inserter(row));
    return row;
}

std::vector<int> ids(const Row& row) {
    std::vector<int> result;
    for (const auto& [member, score] : row) {
        (void)score;
        result.push_back(std::stoi(member));
    }
    return result;
}

// Check membership and scores without relying on rounded floating-point ties.
void expectRow(sw::redis::Redis& redis, int item, const Row& expected) {
    Row actual = readNeighbors(redis, item);
    bool ok = actual.size() == expected.size();
    for (const auto& [member, score] : expected) {
        auto it = std::find_if(actual.begin(), actual.end(),
            [&](const auto& entry) { return entry.first == member; });
        if (it == actual.end() || !std::isfinite(it->second) ||
            std::abs(it->second - score) > 1e-5) {
            ok = false;
        }
    }
    check(ok, "stored similarity row " + std::to_string(item));
    if (!ok) {
        std::cout << "       expected:";
        for (const auto& [member, score] : expected)
            std::cout << " (" << member << ", " << score << ')';
        std::cout << "\n       actual:  ";
        for (const auto& [member, score] : actual)
            std::cout << " (" << member << ", " << score << ')';
        std::cout << '\n';
    }
}

// Cleanup runs on normal return and on exceptions. The dedicated test database
// must remain exclusive to this process throughout the test.
struct Fixture {
    sw::redis::Redis& redis;
    int users;
    int items;

    void clear() {
        redis.del("meta:n");
        redis.del("meta:m");
        for (int u = 0; u < users; ++u)
            redis.del("user:" + std::to_string(u));
        for (int i = 0; i < items; ++i)
            redis.del("top_sim:" + std::to_string(i));
    }

    void seed(const std::vector<Edge>& edges) {
        redis.set("meta:n", std::to_string(users));
        redis.set("meta:m", std::to_string(items));
        for (const auto& [user, item, weight] : edges) {
            redis.zadd("user:" + std::to_string(user),
                       std::to_string(item), weight);
        }
    }

    ~Fixture() {
        try { clear(); }
        catch (const std::exception& ex) {
            std::cerr << "Cleanup failed; inspect the test database: "
                      << ex.what() << '\n';
        }
    }
};

void weightedTests(sw::redis::Redis& redis, const std::string& url) {
    std::cout << "\n=== Weighted similarities and queries ===\n";
    Fixture fixture{redis, 6, 8};
    fixture.seed({
        {0, 0, 1}, {0, 1, 5}, {0, 2, 3},
        {1, 0, 1}, {1, 3, 2},
        {2, 1, 5}, {2, 4, 2},
        {3, 2, 3}, {3, 5, 2},
        // User 4 is empty. User 5's sole item has no co-occurring neighbor.
        {5, 6, 2}
        // Item 7 has no interactions.
    });

    ItemCF cf(url, 3, 4);
    const double s = 1.0 / std::sqrt(2.0);
    expectRow(redis, 0, {{"1", .5}, {"2", .5}, {"3", s}});
    expectRow(redis, 1, {{"0", .5}, {"2", .5}, {"4", s}});
    expectRow(redis, 2, {{"0", .5}, {"1", .5}, {"5", s}});
    expectRow(redis, 3, {{"0", s}});
    expectRow(redis, 4, {{"1", s}});
    expectRow(redis, 5, {{"2", s}});
    expectRow(redis, 6, {});
    expectRow(redis, 7, {});

    expectIds("highest-weight history item; correct Redis item key",
              cf.query(0, 1, 1, 10), {4});
    expectIds("highest-weight two history items", cf.query(0, 2, 1, 10), {4, 5});
    expectIds("weighted candidate ranking", cf.query(0, 3, 1, 10), {4, 5, 3});
    expectIds("result count limit", cf.query(0, 3, 1, 2), {4, 5});
    expectIds("history count above actual size", cf.query(0, 100, 1, 10), {4, 5, 3});
    expectIds("empty user ZSET", cf.query(4, 3, 3, 10), {});
    expectIds("nonempty history with missing similarity ZSET", cf.query(5, 3, 3, 10), {});
    expectIds("all candidates: stored K=3", cf.query(0, 3, 3, 10), {0, 4, 2, 5, 1, 3});
    expectIds("query K above stored K", cf.query(0, 3, 100, 10), {0, 4, 2, 5, 1, 3});

    // All instances share the same Redis model keys: snapshot before rebuilding.
    std::vector<Row> before;
    for (int item = 0; item < 8; ++item) before.push_back(readNeighbors(redis, item));
    std::vector<std::vector<int>> beforeQueries;
    for (int user = 0; user < 6; ++user)
        beforeQueries.push_back(cf.query(user, 3, 3, 10));

    ItemCF singleThread(url, 3, 1);
    for (int item = 0; item < 8; ++item) expectRow(redis, item, before[item]);
    for (int user = 0; user < 6; ++user) {
        expectIds("one vs four threads: user " + std::to_string(user),
                  singleThread.query(user, 3, 3, 10), beforeQueries[user]);
    }

    ItemCF topOne(url, 1, 4);
    expectRow(redis, 0, {{"3", s}});
    expectRow(redis, 1, {{"4", s}});
    expectRow(redis, 2, {{"5", s}});
    expectIds("rebuild with smaller K removes old neighbors",
              topOne.query(0, 3, 100, 10), {4, 5, 3});

    // Keep metadata, remove every user ZSET, and rebuild over the previous model.
    for (int user = 0; user < 6; ++user)
        redis.del("user:" + std::to_string(user));
    ItemCF emptyModel(url, 3, 4);
    bool removed = true;
    for (int item = 0; item < 8; ++item) {
        if (redis.exists("top_sim:" + std::to_string(item)) != 0) removed = false;
    }
    check(removed, "empty rebuild deletes all old similarity keys");
    expectIds("all user ZSETs empty", emptyModel.query(0, 3, 3, 10), {});
}

void tieTests(sw::redis::Redis& redis, const std::string& url) {
    std::cout << "\n=== Exact ties and accumulation ===\n";
    Fixture fixture{redis, 1, 3};
    fixture.seed({{0, 0, 1}, {0, 1, 1}, {0, 2, 1}});
    ItemCF topOne(url, 1, 2);
    expectIds("Top-1 tie: item 0", ids(readNeighbors(redis, 0)), {2});
    expectIds("Top-1 tie: item 1", ids(readNeighbors(redis, 1)), {2});
    expectIds("Top-1 tie: item 2", ids(readNeighbors(redis, 2)), {1});
    expectIds("two seeds contribute to candidate 2", topOne.query(0, 3, 1, 5), {2, 1});

    ItemCF topTwo(url, 2, 2);
    expectIds("equal final scores: numeric ID descending", topTwo.query(0, 3, 2, 5), {2, 1, 0});
    expectIds("final count cutoff", topTwo.query(0, 3, 2, 2), {2, 1});
}

void redisTieTests(sw::redis::Redis& redis, const std::string& url) {
    std::cout << "\n=== Redis string ordering for equal scores ===\n";
    Fixture fixture{redis, 1, 11};
    fixture.seed({{0, 0, 1}, {0, 2, 1}, {0, 10, 1}});
    ItemCF cf(url, 2, 2);
    // Redis compares members as strings: "2" sorts ahead of "10" descending.
    expectIds("equal-similarity Redis order", ids(readNeighbors(redis, 0)), {2, 10});
    expectIds("history tie selects item 2; neighbor tie selects item 10",
              cf.query(0, 1, 1, 5), {10});
    expectIds("final C++ heap uses numeric ID order",
              cf.query(0, 3, 2, 5), {10, 2, 0});
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dedicated-empty-redis-url>\n"
                  << "Example: " << argv[0] << " tcp://127.0.0.1:6379/15\n";
        return 2;
    }

    try {
        const std::string url = argv[1];
        sw::redis::Redis redis(url);
        if (redis.dbsize() != 0) {
            std::cerr << "Refusing to run: the selected Redis database is not empty.\n"
                      << "Use a dedicated empty test database. Existing data was not changed.\n";
            return 2;
        }

        weightedTests(redis, url);
        tieTests(redis, url);
        redisTieTests(redis, url);
        check(redis.dbsize() == 0, "test keys cleaned up");
        std::cout << "\nTOTAL: " << passed << " passed, " << failed << " failed\n";
        return failed == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Redis/ItemCF test aborted: " << ex.what() << '\n';
        return 2;
    }
}