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

Reinforcement learning in RLHF uses the objective:

<img src="http://latex.codecogs.com/gif.latex?\mathcal{L}_{\text{RL}}%20=%20-\mathbb{E}_{(s,%20a,%20r)}%20\left[%20\log%20\pi(a%20\mid%20s)%20\cdot%20\hat{r}%20\right]" />

where $ \hat{r} $ is the reward from a human-trained reward model. However, RLHF faces instability (e.g., PPO), high computational costs, reward hacking, and requires extensive hyperparameter tuning. DPO simplifies this by directly optimising preferences using a stable, maximum likelihood-inspired approach.

Direct Preference Optimization (DPO) reframes preference alignment as a direct likelihood optimisation task. The loss function is:

<img src="http://latex.codecogs.com/gif.latex?\mathcal{L}_{DPO}(\pi\theta;%20\pi_{\text{ref}})%20=%20-\mathbb{E}_{(x,%20y_w,%20y_l)%20\sim%20\mathcal{D}}%20\left[%20\log%20\sigma%20\left(%20\beta%20\log%20\frac{\pi\theta(y_w%20\mid%20x)}{\pi_{\text{ref}}(y_w%20\mid%20x)}%20-%20\beta%20\log%20\frac{\pi\theta(y_l%20\mid%20x)}{\pi_{\text{ref}}(y_l%20\mid%20x)}%20\right)%20\right]" />

Here, $ x $ is the input prompt, $ y_w $ the preferred response, $ y_l $ the less preferred response, and $ \pi_{\text{ref}} $ is the frozen pre-trained reference model. $ \beta $ scales preference weights.

The gradient adjusts probabilities to favour preferred outputs:

<img src="http://latex.codecogs.com/gif.latex?\nabla_\theta%20\mathcal{L}_{DPO}%20=%20\mathbb{E}\left[%20\sigma(...)%20\cdot%20\left(%20\nabla_\theta%20\log%20\pi_\theta(y_w%20\mid%20x)%20-%20\nabla_\theta%20\log%20\pi_\theta(y_l%20\mid%20x)%20\right)%20\right]" />

Unlike RLHF, DPO eliminates explicit reward modelling and reinforcement learning, reducing computational overhead and stabilising training while maintaining alignment effectiveness.


</details>
