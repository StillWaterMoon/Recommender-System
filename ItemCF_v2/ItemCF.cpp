#include "ItemCF.h"

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::size_t h1 = std::hash<int>()(p.first);
        std::size_t h2 = std::hash<int>()(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

using UMP=std::unordered_map<PII,float,PairHash>;

void process_user(const std::vector<PFI>& v,UMP& mp)
{
    int len=v.size();
    for(int i=0;i<len;i++)
    {
        for(int j=i+1;j<len;j++)
        {
            int a=std::min(v[i].second,v[j].second);
            int b=std::max(v[i].second,v[j].second);
            mp[{a,b}]+=v[i].first*v[j].first;
        }
    }
}

void dynamic_process(
    int n,std::atomic<int>& next_user,
    const std::vector<std::vector<PFI>>& e,
    std::vector<UMP>& mps)
{
    while(1)
    {
        int id=next_user.fetch_add(1);
        if(id>=n) break;
        process_user(e[id],mps[id]);
    }
}


ItemCF::ItemCF(std::string redis_url,int k,int cnt_thread):redis(redis_url)
{
    int n = std::stoi(*redis.get("meta:n")),m = std::stoi(*redis.get("meta:m"));
    std::vector<std::vector<PFI>> e(n);
    std::vector<std::vector<PFI>> top_sim(m);
    std::vector<float> sum(m);

    for(int i=0;i<n;i++)
    {
        std::string key = "user:" + std::to_string(i);
        std::vector<std::pair<std::string, double>> items;
        redis.zrevrange(key, 0, -1, std::back_inserter(items));
        for (const auto& [member, score] : items) 
        {
            int item = std::stoi(member);
            float weight = float(score);
            e[i].push_back({weight, item});
            sum[item]+=weight*weight;
        }
    }

    std::vector<std::thread> threads;
    std::vector<std::unordered_map<PII,float,PairHash>> mps(n);
    std::atomic<int> next_user{0};
    for(int i=0;i<cnt_thread;i++)
    {
        threads.push_back(std::thread(dynamic_process,n,std::ref(next_user),std::cref(e),std::ref(mps)));
    }
    for(auto& t:threads)
    {
        t.join();
    }

    UMP global_mp;

    for (const auto& mp : mps) {
        for (const auto& [key, val] : mp) {
            global_mp[key] += val;
        }
    }

    std::vector<std::priority_queue<PFI,std::vector<PFI>,std::greater<PFI>>> q(m);
    for(auto& [key,w]:global_mp)
    {
        int a=key.first,b=key.second;
        float res=w/(std::sqrt(sum[a])*std::sqrt(sum[b]));
        q[a].push({res,b});
        q[b].push({res,a});
        if(q[a].size()>k) q[a].pop();
        if(q[b].size()>k) q[b].pop();
    }
    for(int i=0;i<m;i++)
    {
        while(q[i].size())
        {
            top_sim[i].push_back(q[i].top());
            q[i].pop();
        }
        std::reverse(top_sim[i].begin(),top_sim[i].end());
    }

    auto pipe = redis.pipeline();

    for (int item = 0; item < m; item++) 
    {
        // key = "top_sim:{item}"
        std::string key = "top_sim:" + std::to_string(item);

        // 先删旧的（避免旧数据残留）
        pipe.del(key);

        // 写入新的 Top-K 相似物品
        for (const auto& [sim, sim_item] : top_sim[item]) 
        {
            // ZADD key score member
            pipe.zadd(key,std::to_string(sim_item),sim);
        }
    }

    pipe.exec();
}   

std::vector<int> ItemCF::query(int user_id,int n,int k,int cnt)
{

    std::string key = "user:" + std::to_string(user_id);
    std::vector<std::pair<std::string, double>> items;
    redis.zrevrange(key, 0, n - 1, std::back_inserter(items));

    std::unordered_map<int,float> mp;
    for(int i=0;i<(int)items.size();i++)
    {
        float val=items[i].second;
        int item=std::stoi(items[i].first);

        key = "top_sim:" + std::to_string(item);
        std::vector<std::pair<std::string, double>> top_sim;
        redis.zrevrange(key, 0, k-1, std::back_inserter(top_sim));

        for(int j=0;j<(int)top_sim.size();j++)
        {
            int tmp=std::stoi(top_sim[j].first);
            mp[tmp]+=val*float(top_sim[j].second);
        }
    }
    std::priority_queue<PFI,std::vector<PFI>,std::greater<PFI>> q;
    for(auto& t:mp)
    {
        q.push({t.second,t.first});
        if(q.size()>cnt) q.pop();
    }
    std::vector<int> ans;
    while(q.size()) 
    {
        ans.push_back(q.top().second);
        q.pop();
    }
    std::reverse(ans.begin(),ans.end());
    return ans;
}