<details>
<summary><a href="https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81">Reinforcement Learning for Language Models
</a></summary>

- Supervised learning (SFT) focuses on mimicking demonstrations, limiting generalisation and encouraging memorisation.  
- SFT lacks negative feedback, leading models to hallucinate when answers are unknown.  
- RL enables feedback-driven learning, discouraging falsehoods and encouraging diverse, valid responses.  
- RL supports abstention when the model lacks knowledge, promoting reliability over hallucination.  
- Knowledge-seeking queries benefit from RL by aligning responses with internal knowledge and penalising guessing.  
- RL is technically challenging, requiring effective reward design and scoring mechanisms, often reliant on human feedback.  
- Over-abstention risks exist, requiring careful balance in reward functions.  
- Automating feedback for RL via text-evaluation models is a promising direction to reduce reliance on human input.  
- Hybrid SFT and RL approaches leverage SFT for basic alignment and RL for factual accuracy and abstention.  
- Empirical debates highlight SFT's occasional advantage over RLHF, suggesting nuances in task-specific fine-tuning strategies.  

</details>
<details>
<summary><a href="https://huyenchip.com/2023/05/02/rlhf.html">RLHF – Chip Huyen
</a></summary>

</details>
<details>
<summary><a href="https://arxiv.org/pdf/2305.18290">Direct Preference Optimization:
Your Language Model is Secretly a Reward Model
</a></summary>

In standard RL, the agent updates its policy $\pi(a \mid s)$ to increase the likelihood of actions. e.g. the agent might learn from rewards like $+1$ for a correct action or $0$ for a neutral one.

$$
\mathcal{L}_{\text{RL}} = - \mathbb{E}_{(s, a, r)} \left[ \log \pi(a \mid s) \cdot \hat{r} \right]
$$

where $\hat{r}$ is the reward received after taking action $a$ in state $s$, and the goal is to maximize $\hat{r}$ in the long run.

Direct Preference Optimization (DPO) learns by optimizing pairwise preferences between different outcomes. Instead of using a scalar reward, it trains the model to prefer one outcome over another. The objective is to maximize the likelihood of the preferred outcome being ranked higher than the less-preferred one.

The DPO loss function is given by:

$$
\mathcal{L}_{DPO}(\pi\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$


</details>
