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


ItemCF::ItemCF(int n,int m,int k,std::vector<std::tuple<int,int,float>>& edge,int cnt_thread)
{
    this->n=n,this->m=m,this->k=k;
    e.resize(n),top_sim.resize(m);
    std::vector<float> sum(m);
    for(auto& [u,v,w]:edge)
    {
        e[u].push_back({w,v});
        sum[v]+=w*w;
    }
    for(int i=0;i<n;i++) 
        std::sort(e[i].begin(), e[i].end(), std::greater<PFI>{});
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
}   

std::vector<int> ItemCF::query(int user_id,int n,int k,int cnt)
{
    k=std::min(k,this->k);
    std::unordered_map<int,float> mp;
    for(int i=0;i<std::min(n,(int)e[user_id].size());i++)
    {
        auto& [val,item]=e[user_id][i];
        for(int j=0;j<std::min(k,(int)top_sim[item].size());j++)
        {
            int tmp=top_sim[item][j].second;
            mp[tmp]+=val*top_sim[item][j].first;
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