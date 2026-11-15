from pathlib import Path
import re, zipfile
ROOT=Path(r'F:/KAFA'); OUT=ROOT/'overleaf_kafa_paper_v2'; tex_path=OUT/'main.tex'
text=tex_path.read_text(encoding='utf-8')
abstract=r'''
\begin{abstract}
Federated fine-tuning of large language models (LLMs) is attractive when task data are distributed across private clients, yet it is difficult to deploy because clients often cannot host the same large model as the server. Existing offsite-tuning-style methods, including FedBiOT, demonstrate that LLM adaptation can be performed without exposing the complete server model. KAFA extends this motivation to a broader heterogeneous scenario in which the server maintains a large LLM, clients train smaller LLMs, data are non-IID, supervision may contain irrelevant or corrupted samples, and communication budgets are limited. The core idea is to decouple semantic transfer from parameter transfer. The server compresses task knowledge into a lightweight knowledge anchor consisting of instruction templates, hidden-state prototypes, and soft labels. Clients use the anchor for instruction augmentation, task relevance filtering, representation alignment, and output-distribution distillation. In parallel, a bidirectional projection mapper converts large-model LoRA parameters into small-model LoRA parameters for client training and maps aggregated client updates back to the server model. KAFA therefore forms a closed loop of anchor generation, projected broadcast, anchor-guided local adaptation, prototype-increment upload, and server-side aggregation. This paper presents the formal objective, algorithmic workflow, module-level design, complexity analysis, and an evidence-aware experimental protocol. The visual and empirical artifacts are reported conservatively: verified real logs support pipeline feasibility, whereas synthetic or preflight figures are used only as experiment-design scaffolds until final multi-seed runs are completed.
\end{abstract}
'''
text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda m: abstract.strip(), text, flags=re.S)
intro=r'''
\section{Introduction}
Large language models (LLMs) can be adapted to downstream tasks through supervised fine-tuning, but the data required for such adaptation are frequently private, distributed, and institutionally sensitive. Medical institutions, financial companies, edge-device users, and domain-specific service providers may each hold valuable task data, while none of them can directly share raw records. Federated learning (FL) offers a natural training paradigm: clients optimize locally and a server aggregates updates. However, the direct combination of FL and LLM fine-tuning is not straightforward. The model scale is large, the activation memory is high, and the communication cost of repeatedly moving model updates is substantial. Even when parameter-efficient fine-tuning (PEFT) such as LoRA is used, practical FL still faces a deeper constraint: the server and clients may not own models of the same size.

This work focuses on a heterogeneous LLM fine-tuning setting. The server is assumed to maintain a relatively strong large model, such as a LLaMA-scale model, while clients train smaller LLMs because of GPU memory, latency, deployment, or ownership constraints. This setting is more realistic than the homogeneous assumption that every participant loads an identical model. It is also more difficult. LoRA parameters are layer- and dimension-dependent; if the hidden dimensions of server and client models differ, their LoRA tensors cannot be directly averaged. At the same time, local datasets are non-IID. Clients may differ in instruction style, answer format, reasoning depth, label distribution, and data quality. A client can even contain samples that are unrelated to the target task or whose supervision is corrupted. In such cases, naive aggregation does not merely waste communication; it can actively inject harmful gradients into the global adapter.

FedBiOT provides the main conceptual inspiration for this work. It shows that federated LLM fine-tuning can be designed without giving clients full model access, and it uses public data and bilevel optimization to align server-side and client-side components. KAFA follows the same practical motivation of privacy-preserving and resource-aware LLM adaptation, but it asks a different question. Instead of only compressing or exposing part of the server model, can the server provide a compact semantic coordinate system that remains useful even when clients use genuinely different smaller models? KAFA answers this question through knowledge anchors. A knowledge anchor is not a model copy. It is a task-level abstraction containing instructions, representation prototypes, and soft labels. It transfers what the server believes the task is, rather than exposing the full parameters that implement that belief.

The second half of KAFA is structural. Since semantic anchors alone cannot make LoRA tensors shape-compatible, KAFA introduces a projection mapper between the large-model adapter space and the small-model adapter space. The server down-projects its LoRA parameters before broadcasting them to clients. Clients fine-tune the projected LoRA locally under anchor guidance. Their updates are then aggregated and up-projected back into the large-model LoRA space. This creates a closed semantic-parameter loop: anchors define which local examples are useful and how their representations should behave, while projection defines how trainable updates move across heterogeneous architectures.

The design is deliberately modular. Instruction augmentation reduces prompt-style mismatch. Prototype alignment stabilizes hidden representations. Soft-label distillation transfers output preferences from the server-side task manifold. Relevance filtering and supervision consistency filtering reduce the influence of irrelevant or dirty samples. LoRA projection makes model heterogeneity tractable. Because each module corresponds to a measurable hypothesis, KAFA can be evaluated not only by final accuracy or loss but also by communication efficiency, convergence behavior, robustness, clean-retention rate, dirty-filter precision and recall, and component-level ablation.

The main contributions are summarized as follows.
\begin{itemize}
\item We formulate heterogeneous federated LLM adaptation where the server maintains a large LLM and clients train smaller LLMs under non-IID and potentially noisy supervision.
\item We introduce a knowledge-anchor mechanism that transfers server-side task semantics through instruction templates, prototype banks, and soft labels rather than full-model exposure.
\item We design anchor-guided local training with instruction augmentation, dynamic top-$K$ prototype matching, representation alignment, soft-label distillation, and dirty-supervision consistency filtering.
\item We propose bidirectional LoRA projection to bridge structurally incompatible large-model and small-model adapter spaces.
\item We build an evidence-aware experimental protocol and composite visualization package that reports effectiveness, robustness, communication cost, and evidence credibility without overstating unfinished runs.
\end{itemize}
'''
text=re.sub(r'\\section\{Introduction\}.*?\\section\{Related Work\}', lambda m: intro+'\n\\section{Related Work}', text, flags=re.S)
related_add=r'''
\subsection{Knowledge Anchors as Task-Level Transfer}
A central difference between KAFA and conventional federated parameter averaging is that KAFA transfers a task manifold rather than only an update tensor. Knowledge distillation transfers behavior through output distributions, and prototype learning summarizes semantic regions through representative vectors. KAFA combines both ideas in a federated LLM setting. The prototype bank provides geometric information about where task-relevant samples should lie in representation space, while soft labels provide distributional information about how the server model interprets those regions. This combination is especially useful when client labels are noisy, because the server anchor can act as an independent consistency signal.

\subsection{Robust Federated Training under Dirty Supervision}
Non-IID data are often treated as the main difficulty in FL, but dirty supervision is equally important for LLM fine-tuning. Instruction datasets may contain mismatched answers, incomplete reasoning traces, unrelated examples, or formatting artifacts. If every local example is trusted equally, a small number of corrupted samples can distort local LoRA updates and then spread through aggregation. KAFA therefore evaluates robustness explicitly. The filtering module does not use dirty metadata during training. Instead, it uses relevance to the prototype bank and consistency with soft labels. Dirty labels are used only after the run to measure recall, precision, and clean-retention rate. This distinction is important for a defensible experimental design.
'''
text=text.replace('\\section{Methodology}', related_add+'\n\\section{Methodology}')
method_insert=r'''
\subsection{Design Rationale}
KAFA is based on the observation that heterogeneous federated LLM adaptation contains two separable mismatches. The first is semantic mismatch: a client may hold examples that do not match the intended task, or it may express the task through a different prompt style. The second is structural mismatch: the client model may not share the server model's hidden size or LoRA tensor shapes. Treating these mismatches with one mechanism is fragile. If only parameter projection is used, the client can still learn from irrelevant or corrupted samples. If only semantic distillation is used, the server still cannot aggregate shape-incompatible LoRA updates. KAFA therefore makes the anchor and the projection mapper cooperate. The anchor determines what should be learned; the mapper determines how the learned adapter update can travel.

\subsection{Dirty Supervision Consistency}
For a retained sample, KAFA further checks whether its supervision is consistent with the anchor. Let $\bar{s}_i=\sum_{k\in\mathcal{N}_K(i)}\alpha_{ik}s_k$ be the anchor-induced soft label for sample $i$. If the local label distribution $\ell(y_i)$ sharply conflicts with $\bar{s}_i$, the sample is likely to be corrupted or at least unreliable for global aggregation. A continuous training weight can be written as
\begin{equation}
w_i=\sigma(a(r_i-\eta_r))\sigma(b(c_i-\eta_c)),
\end{equation}
where $r_i$ measures prototype relevance, $c_i=1-D_{KL}(\ell(y_i)\|\bar{s}_i)$ measures supervision consistency, and $\eta_r,\eta_c$ are thresholds. This formulation is more flexible than a hard discard rule. Borderline samples can still contribute weakly, while clearly inconsistent samples have little influence on LoRA gradients.

\subsection{Anchor Refresh and Client Feedback}
The anchor is not necessarily static. After local training, each client can upload incremental prototypes computed from retained samples. The server merges these increments with momentum:
\begin{equation}
p_k^{t+1}=(1-\beta)p_k^t+\beta\Delta p_k^t.
\end{equation}
This update allows the anchor to absorb useful private-task structure without receiving raw text data. To avoid drift, the server can refresh the anchor only when the prototype increment is sufficiently stable across clients or when the validation loss indicates that the current anchor has become stale. This makes the anchor a controlled communication object rather than a one-time prompt engineering artifact.

\subsection{Complexity and Communication Advantage}
The additional computation introduced by KAFA is dominated by prototype matching. For batch size $B$, prototype count $K$, and client anchor dimension $d_c$, naive matching costs $O(BKd_c)$. Since $K$ is small compared with the number of model parameters and the base LLM remains frozen, this overhead is modest relative to transformer forward/backward computation. Communication per round consists of projected LoRA parameters, anchor metadata, and prototype increments. The base model is never transmitted. Compared with RawFedAvg-style dense update exchange, KAFA trades a small amount of structured anchor communication for substantially lower adapter traffic and better robustness instrumentation.
'''
text=text.replace('\\subsection{Algorithm}', method_insert+'\n\\subsection{Algorithm}')
new_prelim=r'''
\section{Preliminary Artifacts and Analysis}
\subsection{Evidence Policy and Current Coverage}
The current experiments are organized with an explicit evidence policy. Verified real runs can support formal claims. Log-only unverified runs may guide debugging or indicate a possible trend, but they should be rerun before final reporting. Synthetic files are used only for plotting, sanity checking, and expected-order visualization. This distinction is intentionally conservative. It protects the final thesis from presenting attractive but insufficiently verified figures as empirical proof.

Table~\ref{tab:artifact} reports the current artifact status. It should be read as a progress table rather than a final leaderboard. The verified GSM8K logs demonstrate that the pipeline can parse training metrics, communication statistics, and dirty-filter behavior. The complete dissertation result matrix should later fill in all tasks, all baselines, and multiple seeds.

\begin{table}[t]
\caption{Current artifact status from parsed logs.}
\label{tab:artifact}
\centering
\small
\begin{tabular}{lccc}
\toprule
Method group & Task & Verified & Unverified \\
\midrule
RawFedAvg & GSM8K & 1 & 1 \\
Ditto & GSM8K & 1 & 1 \\
KAFA-full & GSM8K & 1 & 1 \\
KnowledgeAnchor variants & GSM8K & 3 & 0 \\
Other baselines/ablations & GSM8K & 0 & 1 each \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Enhanced Mechanism Visualization}
Fig.~\ref{fig:loop} complements the original framework diagram with a more abstract semantic-parameter view. It emphasizes that KAFA is not a single communication trick. The server first compresses task knowledge into anchors, then translates trainable LoRA parameters through a projection bridge. Clients receive both objects: the projected adapter gives them a trainable starting point, and the anchor tells them how to select and shape their local updates. This visualization is useful for presentations because it makes the two-channel design immediately visible.

\begin{figure*}[t]
\centerline{\includegraphics[width=0.94\textwidth]{figures/kafa_semantic_parameter_loop.png}}
\caption{Enhanced semantic-parameter view of KAFA. Anchors carry task semantics, while projection carries trainable LoRA structure across heterogeneous model dimensions.}
\label{fig:loop}
\end{figure*}

\subsection{Integrated Evidence Dashboard}
Fig.~\ref{fig:dashboard} replaces isolated single-purpose charts with a composite evidence dashboard. The loss panel shows the expected ordering from synthetic preview files. The communication-performance panel places methods in a two-dimensional space, which is more appropriate for KAFA because the method is designed to improve both adaptation quality and communication cost. The dirty-filtering panel summarizes unit-level corruption probes. The evidence-grade panel makes the reporting discipline explicit. A final version of this dashboard can be regenerated with verified multi-seed results while preserving the same visual language.

\begin{figure*}[t]
\centerline{\includegraphics[width=0.95\textwidth]{figures/experimental_evidence_dashboard.png}}
\caption{Preliminary experimental evidence dashboard. The figure combines loss trends, communication-performance tradeoffs, dirty-filtering metrics, and evidence-grade accounting. Synthetic panels are visualization scaffolds, not final empirical claims.}
\label{fig:dashboard}
\end{figure*}

\subsection{Ablation Anatomy}
Fig.~\ref{fig:ablation} presents ablation as a structural diagnosis. A simple bar chart can show that a variant is worse, but it does not explain which function was removed. The radar panel separates semantic alignment, parameter compatibility, robustness, communication efficiency, and task coverage. The degradation panel shows the expected loss increase when each module is removed. The heatmap links modules to task families. This figure is especially suitable for a thesis defense because it communicates why the modules are necessary rather than only listing their scores.

\begin{figure*}[t]
\centerline{\includegraphics[width=0.95\textwidth]{figures/ablation_anatomy_composite.png}}
\caption{Ablation anatomy of KAFA. Radar, degradation, and heatmap views jointly describe the roles of projection, instruction anchoring, distillation, relevance filtering, and prototype alignment. Values should be replaced by verified ablation runs in the final version.}
\label{fig:ablation}
\end{figure*}

\subsection{Communication and Robustness Views}
Communication efficiency is not only a matter of reporting fewer megabytes. A useful figure should show where the savings come from. Fig.~\ref{fig:comm} compares cumulative communication across representative methods and decomposes the KAFA payload into projected LoRA deltas, anchors, prototype increments, and metadata. This view supports the claim that KAFA's efficiency comes from the combination of frozen base models, projected adapters, and reusable anchors.

\begin{figure*}[t]
\centerline{\includegraphics[width=0.95\textwidth]{figures/communication_efficiency_atlas.png}}
\caption{Communication-efficiency atlas. KAFA exchanges projected LoRA and compact anchor information rather than dense full-model updates.}
\label{fig:comm}
\end{figure*}

Fig.~\ref{fig:filter} gives an intuitive geometry for dirty-supervision filtering. Clean and task-relevant samples lie near prototypes and are retained. Out-of-manifold or inconsistent samples are down-weighted before they can dominate local gradients. The figure is illustrative rather than a direct t-SNE plot, but it clarifies the mechanism behind dirty-filter recall, dirty-filter precision, and clean-retention rate.

\begin{figure*}[t]
\centerline{\includegraphics[width=0.95\textwidth]{figures/anchor_filtering_landscape.png}}
\caption{Knowledge-anchor filtering landscape. The prototype bank defines trusted semantic regions, while low-relevance or inconsistent samples are filtered from local optimization.}
\label{fig:filter}
\end{figure*}
'''
text=re.sub(r'\\section\{Preliminary Artifacts and Analysis\}.*?\\section\{Discussion\}', lambda m: new_prelim+'\n\\section{Discussion}', text, flags=re.S)
discussion=r'''
\section{Discussion}
KAFA separates two difficulties that are often entangled in heterogeneous federated LLM fine-tuning. Structural mismatch is handled by projected LoRA aggregation, while semantic mismatch is handled by knowledge anchors. This design differs from FedBiOT's emulator-adapter optimization: instead of asking clients to train a server-derived compressed component, KAFA allows clients to keep their own smaller model and receive task-level guidance from server-generated anchors. This distinction is important when the client model is not merely a layer-dropped copy of the server model.

The coupling between anchors and projection also explains why the method should be evaluated across multiple dimensions. If KAFA improves loss but costs as much communication as RawFedAvg, it would not satisfy the deployment motivation. If it reduces communication but filters too many clean samples, it would not be robust. If it works only when dirty metadata are available during training, it would not be realistic. For this reason, the experimental protocol includes performance, convergence, communication efficiency, dirty-filter precision and recall, clean-retention rate, nonfinite stability, and evidence credibility.

Several limitations should be acknowledged. First, the quality of the anchor depends on the representativeness of the server-public data. If the public data are too narrow, relevant private samples may be incorrectly considered out-of-manifold. Second, the projection mapper can lose information if the large-small dimension gap is too severe. Third, anchor refresh must be controlled carefully; otherwise the server may absorb noisy client prototypes and shift the task manifold in the wrong direction. These limitations do not weaken the design motivation, but they define the experiments that must be completed before final claims are made.

The current draft deliberately treats incomplete experiments conservatively. Synthetic figures are useful for validating the expected ordering and plotting scripts, but the final thesis should only claim improvements after verified multi-seed runs are available. The most important final tables are the main-result table, ablation table, dirty-filter robustness table, and communication-efficiency table. Once these tables are filled, the current composite figures can be regenerated with verified data and will serve as stronger visual evidence.

\section{Conclusion}
This paper presents KAFA, a knowledge-anchor federated adaptation framework for heterogeneous LLM fine-tuning. KAFA addresses a practical setting in which the server owns a large LLM, clients train smaller LLMs, data are non-IID, supervision may be dirty, and communication is constrained. The method transfers task semantics through instruction templates, prototype banks, and soft labels; trains clients with relevance-weighted prototype alignment and distillation; and bridges heterogeneous LoRA parameters through bidirectional projection. Compared with FedBiOT-inspired offsite tuning, KAFA emphasizes semantic anchoring and large-small adapter compatibility rather than only limited full-model access. The upgraded experimental section reports not just expected performance trends but also robustness, communication cost, component necessity, and evidence grade. Future work will complete the verified multi-task, multi-seed result matrix and replace all synthetic preview figures with final empirical figures.
'''
text=re.sub(r'\\section\{Discussion\}.*?\\balance', lambda m: discussion+'\n\\balance', text, flags=re.S)
# remove old simple figure references if any survived
text=text.replace(r'''\subsection{Preflight Trend Figures}''','')
# add missing references before bibliography end
if 'bibitem{distill}' not in text:
    text=text.replace(
        r'\bibitem{pfedme}',
        r'\bibitem{distill} G. Hinton, O. Vinyals, and J. Dean, ``Distilling the knowledge in a neural network,'' arXiv:1503.02531, 2015.' + '\n' + r'\bibitem{pfedme}'
    )
tex_path.write_text(text, encoding='utf-8')
# zip
zip_path=ROOT/'overleaf_kafa_paper_v2.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        z.write(p,p.relative_to(OUT.parent))
print('wrote', tex_path)
print('words', len(re.findall(r'\b\w+\b', text)))
print('zip', zip_path)


