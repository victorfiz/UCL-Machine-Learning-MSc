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
