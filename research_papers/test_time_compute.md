<details>
<summary><a href="https://arxiv.org/abs/2408.03314">Scaling LLM Test-Time Compute</a></summary>

- Scaling test-time compute (TTC) improves LLM outputs by leveraging additional inference computation adaptively.

- Two primary mechanisms: refining the proposal distribution and employing verifiers to select or refine outputs.

- Proposal refinement involves iterative self-revisions or augmenting prompts to bias outputs; requires task-specific fine-tuning.

- Verifiers can use best-of-N sampling or process-based reward models (PRMs) for stepwise correctness evaluation and tree search.

- Easy tasks benefit from iterative refinements; difficult tasks perform better with parallel sampling or verifier-based search.

Question difficulty guides compute allocation; prompts classified into levels for adaptive TTC scaling.

- FLOPs-matched studies show TTC can outperform a significantly larger model for intermediate tasks.

- Hard tasks still rely on increased pretraining compute for meaningful gains.

– Combining TTC approaches like revisions with PRMs can enhance scaling efficiency but remains underexplored.

- Future work: integrate TTC outputs into training loops, refine difficulty estimation, and develop strategies for harder tasks.

</details>

