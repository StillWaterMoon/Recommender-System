#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <queue>

using PIF=std::pair<int,float>;
using PFI=std::pair<float,int>;
using PII=std::pair<int,int>;

class ItemCF{
public:
    /*
    ItemCF:输入用户数和物品数,以及对应的边
    query:输入用户id,找出top-n的物品，这些物品再找出top-k的相似,共n*k个物品,打分召回top-cnt的物品
    */
    std::vector<std::vector<PFI>> top_sim;
    explicit ItemCF(int n,int m,int k,std::vector<std::tuple<int,int,float>>& edge,int cnt_thread);
    std::vector<int> query(int user_id,int n,int k,int cnt);

private:
    /*
    n:用户数,m:物品数
    e:边,top_sim:每个物品top-k相似
    */
    int n,m,k;
    std::vector<std::vector<PFI>> e;
    // std::vector<std::vector<PFI>> top_sim;
};