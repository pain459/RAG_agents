arXiv:2402.07927v2 [cs.AI] 16 Mar 2025
# A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications

Pranab Sahoo¹, Ayush Kumar Singh¹, Sriparna Saha¹, Vinija Jain²,³∗, Samrat Mondal¹ and Aman Chadha²,³*

# Abstract

Prompt engineering has emerged as an indispensable technique for extending the capabilities of large language models (LLMs) and vision-language models (VLMs). This approach leverages task-specific instructions, known as prompts, to enhance model efficacy without modifying the core model parameters. Rather than updating the model parameters, prompts allow seamless integration of pre-trained models into downstream tasks by eliciting desired model behaviors solely based on the given prompt.

Prompts can be natural language instructions that provide context to guide the model or learned vector representations that activate relevant knowledge. This burgeoning field has enabled success across various applications, from question-answering to commonsense reasoning. However, there remains a lack of systematic organization and understanding of the diverse prompt engineering methods and techniques. This survey paper addresses the gap by providing a structured overview of recent advancements in prompt engineering, categorized by application area. For each prompting approach, we provide a summary detailing the prompting methodology, its applications, the models involved, and the datasets utilized. We also delve into the strengths and limitations of each approach and include a taxonomy diagram and table summarizing datasets, models, and critical points of each prompting technique. This systematic analysis enables a better understanding of this rapidly developing field and facilitates future research by illuminating open challenges and opportunities for prompt engineering.

The significance of prompt engineering is underscored by its capacity to steer model responses, enhancing the adaptability and applicability of LLMs across diverse sectors. The landscape of contemporary prompt engineering spans a spectrum of techniques, encompassing foundational methods like zero-shot and few-shot prompting to more intricate approaches such as "chain of code" prompting. The notion of prompt engineering was initially investigated and popularized in the LLMs [Liu et al., 2023], [Tonmoy et al., 2024], [Chen et al., 2023] later extended to VLMs [Wu et al., 2023], [Bahng et al., 2022]. Despite the extensive literature on prompt engineering within both LLMs and VLMs, a notable gap remains, particularly concerning a systematic overview of application-centric prompt engineering techniques. With recent strides in prompt engineering, there is a pressing need for a comprehensive survey that offers a nuanced understanding of applications and advancements in contemporary research.

# 1 Introduction

Prompt engineering has emerged as a crucial technique for enhancing the capabilities of pre-trained large language models (LLMs) and vision-language models (VLMs). It involves strategically designing task-specific instructions, referred to as prompts, to guide model output without altering parameters. The significance of prompt engineering is especially evident in its transformative impact on the adaptability of LLMs and VLMs. By offering a mechanism to fine-tune model outputs through carefully crafted instructions, prompt engineering enables these models to excel across diverse tasks and domains. This adaptability is different from traditional paradigms, where model retraining or extensive fine-tuning is often required for task-specific performance. This is the transformative promise of prompt engineering, pushing the boundaries of AI and opening doors to a future brimming with possibilities. In an ever-evolving landscape, ongoing research consistently reveals innovative approaches and applications within prompt engineering. This survey dives deep into the ever-evolving landscape of prompt engineering, analyzing over 41 distinct techniques categorized by their diverse applications. Employing a systematic approach, we aim to provide a comprehensive overview of the current state of prompt engineering.

∗Work does not relate to position at Amazon.

---


review approach, we meticulously delve into the intricacies

# 2.2 Reasoning and Logic

Chain-of-Thought (CoT) Prompting

LLMs often stumble in the face of complex reasoning, limiting their potential. Aiming to bridge this gap, Wei et al. [2022] introduced Chain-of-Thought (CoT) prompting as a technique to prompt LLMs in a way that facilitates coherent and step-by-step reasoning processes. The primary contribution lies in the proposal and exploration of CoT prompting, demonstrating its effectiveness in eliciting more structured and thoughtful responses from LLMs compared to traditional prompts. Through a series of experiments, the authors showcase the distinctive qualities of CoT prompting, emphasizing its ability to guide LLMs through a logical reasoning chain. This results in responses that reflect a deeper understanding of the given prompts. For example, the prompt would show the reasoning process and final answer for a multi-step math word problem and mimic how humans break down problems into logical intermediate steps. The authors achieved state-of-the-art performance in math and commonsense reasoning benchmarks by utilizing CoT prompts for PaLM 540B, achieving an accuracy of 90.2%.

Automatic Chain-of-Thought (Auto-CoT) Prompting

Manual creation of high-quality CoT examples is both time-consuming and suboptimal. Zhang et al. [2022] introduced Auto-CoT to automatically instruct LLMs with a "Let’s think step-by-step" prompt to generate reasoning chains. Recognizing the possibility of errors in individually generated chains, Auto-CoT enhances robustness through diverse sampling. It samples various questions and generates multiple distinct reasoning chains for each, forming a final set of demonstrations. This automated diverse sampling minimizes errors and enhances few-shot learning, eliminating the need for labor-intensive manual creation of reasoning chains. Auto-CoT demonstrated enhanced performance, surpassing the CoT paradigm with average accuracy improvements of 1.33% and 1.5% on arithmetic and symbolic reasoning tasks, respectively, employing GPT-3.

# 2 Prompt Engineering

In this section, we have organized prompt engineering techniques according to their application areas and provided a concise overview of the evolution of prompting techniques, spanning from zero-shot prompting to the latest advancements.

# 2.1 New Tasks Without Extensive Training

Zero-Shot Prompting

Zero-shot prompting offers a paradigm shift in leveraging large LLMs. This technique removes the need for extensive training data, instead relying on carefully crafted prompts that guide the model toward novel tasks [Radford et al., 2019]. Specifically, the model receives a task description in the prompt but lacks labeled data for training on specific input-output mappings. The model then leverages its pre-existing knowledge to generate predictions based on the given prompt for the new task.

# Few-Shot Prompting

Few-shot prompting provides models with a few input-output examples to induce an understanding of a given task, unlike zero-shot prompting, where no examples are supplied [Brown et al., 2020]. Providing even a few high-quality examples has improved model performance on complex tasks compared to no demonstration. However, few-shot prompting requires additional tokens to include the examples, which may become prohibitive for longer text inputs. Moreover, the selection and composition of prompt examples can significantly influence model behavior, and biases like favoring frequent words may still affect few-shot results. While few-shot prompting enhances capabilities for complex tasks, especially among large pre-trained models like GPT-3, careful prompt engineering is critical to achieve optimal performance and mitigate unintended model biases.

# Logical Chain-of-Thought (LogiCoT) Prompting

The ability to perform logical reasoning is critical for LLMs to solve complex, multi-step problems across diverse domains. Existing methods, like CoT prompting, encourage step-by-step reasoning.



---


# New Tasks Without Extensive Training

# §2.1

- Zero-shot Prompting [Radford et al., 2019]
- Few-shot Prompting [Brown et al., 2020]
- Chain-of-Thought (CoT) Prompting [Wei et al., 2022]
- Automatic Chain-of-Thought (Auto-CoT) [Zhang et al., 2022]
- Self-Consistency [Wang et al., 2022]
- Logical CoT (LogiCoT) Prompting [Zhao et al., 2023]
- Chain-of-Symbol (CoS) Prompting [Hu et al., 2023]
- Tree-of-Thoughts (ToT) Prompting [Yao et al., 2023a]
- Graph-of-Thought (GoT) Prompting [Yao et al., 2023b]
- System 2 Attention Prompting [Weston and Sukhbaatar, 2023]
- Thread of Thought (ThoT) Prompting [Zhou et al., 2023]
- Chain of Table Prompting [Wang et al., 2024]
- Self-Refine Prompting [Madaan et al., 2023]

# Reasoning and Logic

# §2.2

- Code Prompting [Puerto et al., 2024]
- Self-Harmonized CoT (ECHO) Prompting [Mekala et al., 2024]
- Logic-of-Thought Prompting [Liu et al., 2024]
- Instance-adaptive Prompting (IAP) [Yuan et al., 2024]
- End-to End DAG-Path (EEDP) Prompting [Yuan et al., 2024]
- Layer-of-Thoughts (LoT) [Fungwacharakorn et al., 2024]
- Narrative-of-Thought (NoT) Prompting [Zhang et al., 2024]
- Buffer of Thoughts (BoT) Prompting [Yang et al., 2024]

# Prompt Engineering

- Contrastive Denoising with Noisy Chain-of-Thought (CD-CoT) Prompting [Zhou et al., 2024]
- Reverse Chain-of-Thought (R-CoT) Prompting [Deng et al., 2024]
- Chain of Draft (CoD) Prompting [Xu et al., 2025]
- Retrieval Augmented Generation (RAG) [Lewis et al., 2020]
- ReAct Prompting [Yao et al., 2022]

# Reduce Hallucination

# §2.3

- Chain-of-Verification (CoVe) [Dhuliawala et al., 2023]
- Chain-of-Note (CoN) Prompting [Yu et al., 2023]
- Chain-of-Knowledge (CoK) Prompting [Li et al., 2023d]

# User Interaction

# §2.4

- Active-Prompt [Diao et al., 2023]

# Fine-Tuning and Optimization

# §2.5

- Automatic Prompt Engineer (APE) [Zhou et al., 2022]

# Knowledge-Based Reasoning and Generation

# §2.6

- Automatic Reasoning and Tool-use (ART) [Paranjape et al., 2023]

# Improving Consistency and Coherence

# §2.7

- Contrastive Chain-of-Thought Prompting (CCoT) [Chia et al., 2023]

# Managing Emotions and Tone

# §2.8

- Emotion Prompting [Li et al., 2023a]
- Scratchpad Prompting [Nye et al., 2021]
- Program of Thoughts (PoT) Prompting [Chen et al., 2022]

# Code Generation and Execution

# §2.9

- Structured Chain-of-Thought (SCoT) Prompting [Li et al., 2023c]
- Chain of Code (CoC) Prompting [Li et al., 2023b]

# Optimization and Efficiency

# §2.10

- Optimization by Prompting [Yang et al., 2023]

# Understanding User Intent

# §2.11

- Rephrase and Respond (RaR) Prompting [Deng et al., 2023]

# Metacognition and Self-Reflection

# §2.12

- Take a Step Back Prompting [Zheng et al., 2023]

# Figure 2

Taxonomy of prompt engineering techniques in LLMs, organized around application domains, providing a nuanced framework for customizing prompts across diverse contexts.



---


# Reasoning Frameworks in Language Models

Zhao et al. [2023] proposes a Logical Chain-of-Thought (LogiCoT) prompting, a neurosymbolic framework that leverages principles from symbolic logic to enhance reasoning in a coherent and structured manner. Specifically, LogiCoT applies the concept of reductio ad absurdum to verify each step of reasoning generated by the model and provide targeted feedback to revise incorrect steps. LogiCoT can reduce logical errors and hallucinations through a think-verify-revise loop. Experimenting with Vicuna-33b and GPT-4, the findings underscore LogiCoT’s notable enhancement of reasoning abilities, exhibiting improvements of 0.16% and 1.42% on the GSM8K dataset and 3.15% and 2.75% on the AQuA dataset compared to CoT, respectively.

# System 2 Attention (S2A) Prompting

The soft attention mechanism in Transformer-based LLMs is prone to incorporating irrelevant context information, impacting token generation adversely. To address this, Weston and Sukhbaatar [2023] proposed System 2 Attention (S2A), utilizing the reasoning abilities of LLMs to selectively attend to relevant portions by regenerating the input context. S2A employs a two-step process to enhance attention and response quality by employing context regeneration and response generation with refined context. The effectiveness of S2A is evaluated across various tasks, including factual QA, long-form generation, and math word problems. In factual QA, S2A attains an accuracy of 80.3%, demonstrating a substantial enhancement in factuality. In long-form generation, it improves objectivity and receives a score of 3.82 out of 5.

# Thread of Thought (ThoT) Prompting

Zhou et al. [2023] presented Thread of Thought (ThoT), a prompting technique designed to enhance the reasoning abilities of LLMs within chaotic contexts. ThoT, inspired by human cognition, systematically examines extensive contexts into manageable segments for incremental analysis, employing a two-phase approach where the LLM first summarizes and examines each segment before refining the information for a final response. ThoT’s flexibility shines as a versatile "plug-and-play" module, enhancing reasoning across different models and prompting methods. Evaluations on question answering and conversation datasets reveal substantial performance improvements of 47.20% and 17.8%, respectively, especially in chaotic contexts.

# Chain-of-Table Prompting

Approaches like CoT, PoT, and ToT represent reasoning steps through free-form text or code, which face challenges when dealing with intricate table scenarios. The study by Wang et al. [2024] introduced a pioneering prompting technique named Chain-of-Table. This method uses step-by-step tabular reasoning by dynamically generating and executing common SQL/DataFrame operations on tables. The iterative nature of this process enhances intermediate results, empowering LLMs to make predictions through logically visualized reasoning chains. Significantly, Chain-of-Table consistently improves the performance of two benchmark tabular datasets by 8.69% on TabFact and 6.72% on WikiTQ, respectively.

# Self-Refine Prompting

Self-Refine prompting, proposed by Madaan et al. [2023], enhances LLM performance by iteratively refining outputs.



---

through self-generated feedback, mimicking human revision. in Mixtral-8x7B, though it remained behind GPT-3.5, a gap attributed to differences in the quality of reasoning rationales.

# Logic-of-thought Prompting

LLMs often exhibit unfaithful reasoning, where the generated conclusions diverge from the intermediate reasoning steps. Logic-of-Thought prompting [Liu et al., 2024] is a neuro-symbolic framework developed to mitigate this issue by enriching prompts with logical information derived from propositional logic. LoT operates in three phases: (1) Logic Extraction, during which LLMs identify propositions and logical relationships from input texts; (2) Logic Extension, in which a Python-based module applies formal logical laws (e.g., contraposition) to infer additional expressions; and (3) Logic Translation, where the extended logic is rendered back into natural language and appended to the original prompt to ensure contextual fidelity. Moreover, Logic-of-thought is designed to integrate seamlessly with other prompting strategies such as CoT, Self-Consistency, and ToT prompting. Reported evaluations indicate that Logic-of-thought can improve CoT accuracy on the ReClor benchmark by 4.35%, enhance CoT prompting with Self-Consistency on LogiQA by 5%, and further boost ToT prompting performance on the ProofWriter dataset by 8%. Additionally, by preserving natural language representations throughout the process, Logic-of-Thought avoids the symbolic extraction errors that can impair other neuro-symbolic systems, such as SatLM.

# Code Prompting

Pre-training on code enhances the reasoning capabilities of LLMs, yet the underlying mechanisms driving this improvement remain poorly understood. To investigate this, Puerto et al. [2024] examines the impact of input representation on LLM reasoning, specifically exploring whether reformulating natural language (NL) problems into code can trigger conditional reasoning abilities. This led to the introduction of Code Prompting, a technique that reformulates NL tasks into structured code, enabling direct prompting of text+code LLMs without relying on external code execution. Experiments on three reasoning benchmarks, ConditionalQA, BoardgameQA, and ShARC, demonstrate that code prompts significantly outperform traditional text-based prompts. On average, GPT 3.5 achieved a performance gain of 8.42 F1 score, while Mistral showed an average improvement of 4.22 across the three datasets.

# Self-Harmonized Chain-of-Thought (ECHO) Prompting

While Chain-of-Thought prompting enhances reasoning in LLMs, methods like Auto-CoT, which automate demonstration generation, face challenges from misleading similarity (incorrect rationales in similar examples) and ineffective diversity (irrelevant or overly varied patterns). To address these issues, Mekala et al. [2024] introduced ECHO, a self-harmonized prompting framework that unifies diverse reasoning paths into a coherent pattern, balancing automation with robustness. ECHO operates through three key stages: (1) Question Clustering, where Sentence-BERT embeddings and k-means group questions into clusters; (2) Demonstration Sampling, which selects representative questions from each cluster and generates rationales using Zero-Shot-CoT; and (3) Demonstration Unification, where rationales are iteratively refined using a dynamic prompting mechanism to align reasoning patterns. This process minimizes diversity-induced noise while retaining adaptability. ECHO surpassed Auto-CoT by an average of 2.8% across 10 reasoning benchmarks (arithmetic, commonsense, symbolic) while demonstrating greater efficiency. It retained performance with 50% fewer examples, showing only a -0.8% dip compared to Few-Shot-CoT’s -1.3% decline. The method also achieved 2.3% gains over Auto-CoT.

# End-to End DAG-Path (EEDP) Prompting

End-to-End DAG-Path (EEDP) prompting [Hong et al., 2024] addresses the limitations of traditional graph-flattening methods, such as adjacency lists and edge lists, which struggle with long-distance reasoning in graph-related tasks for LLMs. EEDP’s key insight is that conventional flattened representations often lose critical long-range dependencies.

---


essential for effective reasoning. To mitigate this, EEDP requiring additional model training. NOT comprises three core components: (1) Structural Representation, where events are encapsulated in a Python class and processed through code completion; (2) NOT Prompting template, which generates temporally grounded narratives to guide the construction of temporal graphs; and (3) Narrative-Aware Demonstrations, utilizing GPT-4-generated few-shot examples optimized for both conciseness and accuracy. Results demonstrated that NOT significantly improves the performance of small LLMs, with LLaMA3-8B achieving an F1 score of 42.2, closely matching GPT-3.5’s 45.7, while exhibiting superior structural coherence.

EEDP was evaluated on tasks such as Edge Prediction Connectivity Prediction (EPCP) and Edge Prediction Distance Prediction (EPDP) using educational (Merged_1000) and molecular (ZINC_test_2500) datasets. The evaluation results highlighted significant performance gains over traditional baselines, with EPCP showing a +10.21% accuracy improvement on Merged_1000 and +16.76% on ZINC_test_2500. Similarly, EPDP achieved a +4.73% accuracy boost on Merged_1000 and an impressive +30.13% on ZINC_test_2500.

# Buffer of Thoughts (BoT) Prompting

Existing prompting methods often struggle to balance universality, efficiency, and robustness in complex reasoning. To address this, Yang et al. [2024] introduced Buffer of Thoughts (BoT), a framework that enhances LLMs through reusable high-level reasoning patterns. BoT overcomes the limitations of single-query methods (e.g., manual exemplar reliance) and multi-query approaches (e.g., computational inefficiency) by introducing a meta-buffer that distills "thought-templates" from diverse tasks and a dynamic buffer-manager that continuously refines them as new problems are solved.

BoT retrieves task-specific thought-templates (e.g., structured problem-solving approaches) and adaptively instantiates them, mimicking human analogical reasoning to eliminate manual prompt design and recursive exploration. Experiments across 10 benchmarks demonstrate its state-of-the-art performance, achieving gains of 11% on Game of 24, 20% on Geometric Shapes, and 51% on Checkmate-in-One, while using just 12% of the computational cost of multi-query methods like Tree-of-Thoughts. Notably, BoT enhances smaller models, with Llama3-8B + BoT surpassing Llama3-70B in accuracy, showing its potential to democratize efficient reasoning at scale.

# Contrastive Denoising with Noisy Chain-of-Thought (CD-CoT) Prompting

Contrastive Denoising with Noisy Chain-of-Thought (CD-CoT) [Zhou et al., 2024] addresses the challenge of "noisy rationales" in chain-of-thought prompting, where irrelevant or incorrect intermediate reasoning steps degrade LLM performance. The NoRa (Noisy Rationales) dataset highlights this issue, showing that LLMs often perform worse with flawed rationales than with no examples at all, as they tend to mimic incorrect reasoning. Existing methods like self-correction and self-consistency offer limited solutions, as self-correction fails without external feedback, and self-consistency selects frequent answers without resolving reasoning flaws.

CD-CoT mitigates this by contrasting noisy rationales with clean ones, rephrasing flawed examples, selecting optimal reasoning paths, and voting on the most consistent answer. Experiments show that CD-CoT improves accuracy by 17.8% on average, significantly outperforming baselines and enhancing LLMs’ robustness in reasoning-intensive tasks.

# Reverse Chain-of-Thought (R-CoT) Prompting

Deng et al. [2024] introduced the Reverse Chain-of-Thought (R-CoT) pipeline, a novel approach to enhancing geometric reasoning in LMMs by addressing dataset limitations such as low quality, diversity, and fidelity. R-CoT operates in two...



---


# ReAct Prompting

Unlike previous studies that treated reasoning and action separately, ReAct [Yao et al., 2022] enables LLMs to generate reasoning traces and task-specific actions concurrently. This interleaved process enhances synergy between reasoning and action, facilitating the model in inducing, tracking, and updating action plans while handling exceptions. ReAct is applied to diverse language and decision-making tasks, showcasing its effectiveness over state-of-the-art baselines. Notably, in question answering (HotpotQA) and fact verification (Fever), ReAct addresses hallucination and error propagation issues by interacting with a simple Wikipedia API, producing more interpretable task-solving trajectories. Additionally, in interactive decision-making benchmarks like ALFWorld and WebShop, ReAct surpasses both imitation and reinforcement learning approaches, achieving notable success rates of 34% and 10%, respectively, with minimal in-context examples.

# Chain of Draft (CoD) Prompting

Chain of Draft (CoD) [Xu et al., 2025], a novel prompting strategy designed to enhance efficiency in complex reasoning tasks. Unlike traditional CoT prompting, which emphasizes detailed step-by-step reasoning, CoD generates concise, information-dense outputs at each step, mirroring human problem-solving strategies where only essential insights are noted. While CoT improves reasoning accuracy, it often leads to verbose outputs and increased computational costs. CoD mitigates this by constraining word usage in each reasoning step, reducing latency and token consumption without sacrificing accuracy. This efficiency-oriented approach is particularly valuable for real-world applications where computational resources and response time are critical. Experimental results across arithmetic, commonsense, and symbolic reasoning benchmarks demonstrate that CoD matches or even outperforms CoT in accuracy while significantly lowering token usage and latency. In some cases, CoD achieved comparable accuracy with an 80% reduction in output tokens as well as an average latency reduction of 76.2%, demonstrating its potential as a lightweight yet effective alternative to traditional prompting strategies.

# Chain-of-Verification (CoVe) Prompting

To address hallucinations in LLMs, Dhuliawala et al. [2023] proposed Chain-of-Verification (CoVe), which involves a systematic four-step process including the model generate baseline responses, plan verification questions to check its work, answer the questions independently, and produce a revised response incorporating the verification. By verifying its work through this deliberate multi-step approach, the LLM enhances logical reasoning abilities and reduces errors even with contradictory information. CoVe emulates human verification to bolster the coherence and precision of LLM output. Experiments on list questions, QA, and long-form generation demonstrate that CoVe decreases hallucinations while maintaining facts [Sahoo et al., 2024]. Focused verification questions help models identify and correct their inaccuracies.

# Chain-of-Note (CoN) Prompting

Retrieval-augmented language models (RALMs) enhance large language models by incorporating external knowledge to reduce factual hallucination. However, the reliability of retrieved information is not guaranteed, leading to potentially misguided responses. Standard RALMs struggle to assess their knowledge adequacy and often fail to respond with "unknown" when lacking information. To address these challenges, Yu et al. [2023] introduced a novel approach to improve RALMs robustness by handling noisy, irrelevant documents and accurately addressing unknown scenarios. CoN systematically evaluates document relevance, emphasizing critical and reliable information to filter out irrelevant content, resulting in more precise and contextually relevant responses. Testing across diverse open-domain question-answering datasets demonstrated notable improvements, including a +7.9 average boost in exact match scores for noisy retrieved documents and a +10.5 enhancement in rejection rates for questions beyond pre-training knowledge.

# Chain-of-Knowledge (CoK) Prompting

Traditional prompting techniques for LLMs have proven powerful in tackling basic tasks. However, their efficacy diminishes due to complex reasoning challenges, often resulting in unreliable outputs plagued by factual hallucinations and opaque thought processes. This limitation arises from their reliance.



---


on fixed knowledge sources, ineffective structured query generation, and lack of progressive correction that fails to guide the LLM adequately. Motivated by human problem-solving, CoK [Li et al., 2023d] systematically breaks down intricate tasks into well-coordinated steps. The process initiates with a comprehensive reasoning preparation stage, where the context is established, and the problem is framed. Subsequently, it engages in a dynamic knowledge adaptation phase, meticulously gathering evidence from various sources, such as its internal knowledge base, external databases, and the given prompt.

# 2.4 User Interface

# Active Prompting

Diao et al. [2023] introduced Active-Prompting as a solution to the challenge of adapting LLMs to diverse reasoning tasks. They address the issue by proposing Active-Prompt to enhance LLMs’ performance on complex question-and-answer tasks through task-specific example prompts with chain-of-thought (CoT) reasoning. Unlike existing CoT methods that rely on fixed sets of human-annotated exemplars, Active-Prompt introduces a mechanism for determining the most impactful questions for annotation. Drawing inspiration from uncertainty-based active learning, the method utilizes various metrics to characterize uncertainty and selects the most uncertain questions for annotation. Active-Prompting exhibits superior performance, outperforming self-consistency by an average of 7.0% and 1.8% across eight complex reasoning tasks in text-davinci-002 and code-davinci-002, respectively, showcasing state-of-the-art results.

# 2.5 Fine-Tuning and Optimization

# Automatic Prompt Engineer (APE)

While crafting effective prompts for LLMs has traditionally been a laborious task for expert annotators, Zhou et al. [2022] introduced Automatic Prompt Engineer (APE) as an innovative approach to automatic instruction generation and selection for LLMs. APE sheds the limitations of static, hand-designed prompts by dynamically generating and selecting the most impactful prompts for specific tasks. This ingenious method analyzes user input, crafts candidate instructions, and then leverages reinforcement learning to choose the optimal prompt, adapting it on the fly to different contexts. Extensive tests on the diverse BIG-Bench suite and the CoT reasoning task revealed APE’s prowess, exceeding human-authored prompts in most cases (19 out of 24 tasks) and significantly boosting LLMs reasoning abilities.

# 2.6 Knowledge-Based Reasoning and Generation

# Automatic Reasoning and Tool-use (ART)

The limited reasoning abilities and lack of external tool utilization hinder the potential of LLMs in complex tasks. Paranjape et al. [2023] introduced Automatic Reasoning and Tool-use (ART) to tackle this critical barrier that empowers LLMs to reason through multi-step processes and seamlessly integrate external expertise. ART bridges the reasoning gap, enabling LLMs to tackle complex problems and expand beyond simple.

# 2.7 Improving Consistency and Coherence

# Contrastive Chain-of-Thought (CCoT) Prompting

Traditional CoT prompting for LLMs often misses a crucial element: learning from mistakes. That is where Contrastive Chain-of-Thought Prompting (CCoT) [Chia et al., 2023] dives in, providing both valid and invalid reasoning demonstrations alongside original prompts. Imagine exploring a map with the right path and the wrong turns to avoid – that is the advantage of contrastive CoT! This dual-perspective approach, tested on reasoning benchmarks like SQuAD and COPA, pushes LLMs to step-by-step reasoning, leading to 4-16% improvements in strategic and mathematical reasoning evaluations compared to traditional CoT, further improved by approximately 5% when integrated with self-consistency techniques. However, questions remain about this technique, such as the automated generation of contrasting demonstrations for diverse problems and its applicability to other NLP tasks beyond reasoning.

# 2.8 Managing Emotions and Tone

# Emotion Prompting

While LLMs demonstrate impressive capabilities on various tasks, their ability to comprehend psychological and emotional cues remains uncertain. The study by Li et al. [2023a] addressed the uncertainty surrounding LLMs’ ability to comprehend emotional cues by introducing EmotionPrompt. Drawing inspiration from psychological research on language’s impact on human performance, they append 11 emotional stimulus sentences to prompts to enhance LLM emotional intelligence. Experimental results demonstrate seamless integration of these stimuli, significantly improving LLM performance across various tasks. EmotionPrompt demonstrates an 8.00% relative improvement in instruction induction and an impressive 115% boost in BIG-Bench tasks, underscoring its efficacy in augmenting LLM capabilities in processing affective signals. An evaluation involving 106 participants reveals an average improvement of 10.9% in performance, truthfulness, and responsibility metrics for generative tasks when employing EmotionPrompt compared to standard prompts.

# 2.9 Code Generation and Execution

# Scratchpad Prompting

Despite the prowess of Transformer-based language models in generating code for basic programming tasks, they encounter



---


# 2.10 Optimization and Efficiency

Optimization by Prompting (OPRO)

In various domains, optimization is a fundamental process often involving iterative techniques. Yang et al. [2023] introduce Optimization by PROmpting (OPRO), a novel approach that leverages LLMs as optimizers. Unlike traditional methods, OPRO utilizes natural language prompts to iteratively generate solutions based on the problem description, enabling quick adaptation to different tasks and customization of the optimization process. The potential of LLMs for optimization is demonstrated through case studies on classic problems like linear regression and the traveling salesman problem. Additionally, it explores the optimization of prompts to maximize accuracy in natural language processing tasks, highlighting the sensitivity of LLMs. The experiments show that optimizing prompts for accuracy on a small training set effectively translates to high performance on the test set. OPRO leads to a significant performance boost, with the most effective prompts optimized by OPRO outperforming human-designed prompts by up to 8% on the GSM8K dataset and up to 50% on challenging tasks in Big-Bench.

# 2.11 Understanding User Intent

Rephrase and Respond (RaR) Prompting

The study by Deng et al. [2023] brings attention to an often-neglected dimension in exploring LLMs: the disparity between human thought frames and those of LLMs and introduces Rephrase and Respond (RaR). RaR allows LLMs to rephrase and expand questions in a single prompt, demonstrating improved comprehension and response accuracy. The two-step RaR variant, incorporating rephrasing and response LLMs, achieves substantial performance enhancements across various tasks. The study highlights that in contrast to casually posed human queries, the rephrased questions contribute to enhanced semantic clarity and the resolution of inherent ambiguity. These findings offer valuable insights for understanding and enhancing the efficacy of LLMs across various applications.

# 2.12 Metacognition and Self-Reflection

Take a Step Back Prompting

Addressing the persistent challenge of complex multi-step reasoning, Zheng et al. [2023] introduced the Step-Back prompting technique, tailored explicitly for advanced language models like PaLM-2L. This innovative approach empowers models to engage in abstraction, extracting high-level concepts and fundamental principles from specific instances. The Step-Back prompting method involves a two-step procedure, integrating Abstraction and Reasoning. Through extensive experiments, applying Step-Back Prompting to PaLM-2L in diverse reasoning-intensive tasks such as STEM, Knowledge QA, and Multi-Hop Reasoning, the results demonstrate a substantial enhancement in reasoning capabilities. Noteworthy performance boosts are observed, with improvements in tasks like MMLU Physics and Chemistry by 7%, TimeQA by 27%, and MuSiQue by 7%.



---



<table>
Table 1: Summary of prevalent prompting techniques of LLMs based on the following factors: application, prompt acquisition, prompt turn, language model, dataset, and metrics.
<thead><tr> <th>Application</th>
<th>Prompting Technique</th>
<th>Prompt Acquisition</th>
<th>Prompt Turn</th>
<th>Language Model(s)</th>
<th>Dataset</th>
<th>Metric(s)</th>
</tr>
</thead>
<tbody>
<tr>
<td>New Tasks Without Training Data</td>
<td>Zero-shot</td>
<td>Manual</td>
<td>Single</td>
<td>GPT-2</td>
<td>Arithmetic, Symbolic</td>
<td>Accuracy, ROUGE Score</td>
</tr>
<tr>
<td></td>
<td>Few-shot</td>
<td>Manual</td>
<td>Single</td>
<td>GPT-3</td>
<td>NaturalQS, WebQS, TriviaQA</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>CoT</td>
<td>Manual</td>
<td>Multi</td>
<td>PaLM 540B</td>
<td>GSM8K</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>LogiCoT</td>
<td>Manual</td>
<td>Multi</td>
<td>Vicuna-33b, GPT-4</td>
<td>GSM8K, AQuA, SocialQA</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>CoS</td>
<td>Manual</td>
<td>Multi</td>
<td>gpt-3.5-turbo, GPT-4</td>
<td>SPARTUN</td>
<td>Accuracy, Precision, Recall</td>
</tr>
<tr>
<td></td>
<td>Auto-CoT</td>
<td>LM Generated</td>
<td>Multi</td>
<td>GPT-3</td>
<td>Arithmetic, Symbolic</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>Self-Consistency</td>
<td>Manual</td>
<td>Single</td>
<td>PaLM 540B</td>
<td>Arithmetic, Commonsense</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>ToT</td>
<td>Retrieval Based</td>
<td>Multi</td>
<td>GPT-4</td>
<td>Game of 24, Creative Writing</td>
<td>Success Rate</td>
</tr>
<tr>
<td></td>
<td>GoT</td>
<td>Retrieval Based</td>
<td>Multi</td>
<td>T5-large</td>
<td>GSM8K, ScienceQA</td>
<td>ROUGE Score</td>
</tr>
<tr>
<td></td>
<td>S2A</td>
<td>Manual</td>
<td>Single</td>
<td>Llama 2-70B</td>
<td>QA, GSM8K</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>ThoT</td>
<td>Hybrid</td>
<td>Multi</td>
<td>gpt-3.5-turbo, Llama 2-70b-chat</td>
<td>PopQA, EntityQ, MTCR</td>
<td>Exact Match (EM) Score</td>
</tr>
<tr>
<td></td>
<td>Chain of Table</td>
<td>Manual</td>
<td>Multi</td>
<td>GPT 3.5, LLaMA 2</td>
<td>TabFact, WikiTQ</td>
<td>BLEU, ROUGE Score</td>
</tr>
<tr>
<td>Reasoning and Logic</td>
<td>Self-Refine</td>
<td>Manual</td>
<td>Multi</td>
<td>GPT-3.5, GPT-4</td>
<td>7 diverse tasks (e.g., Dialogue Response, Math Reasoning)</td>
<td>Task-specific (Accuracy, Human Preference)</td>
</tr>
<tr>
<td></td>
<td>Code Prompting</td>
<td>LM Generated</td>
<td>Multi</td>
<td>GPT 3.5, Mixtral</td>
<td>CondQA, ShaRC, BGQA</td>
<td>F1</td>
</tr>
<tr>
<td></td>
<td>ECHO</td>
<td>Hybrid</td>
<td>Multi</td>
<td>gpt-3.5-Turbo-0301</td>
<td>Arithmetic, Commonsense, Symbolic</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>Logic-of-thought</td>
<td>LM Generated</td>
<td>Multi</td>
<td>GPT 3.5-turbo, GPT-4</td>
<td>ReClor, LogiQA, RuleTaker, ProofWriter, FOLIO</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>CoD</td>
<td>Hybrid</td>
<td>Single</td>
<td>GPT-4o, Claude 3.5 Sonnet</td>
<td>Arithmetic, Commonsense, Symbolic</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>CoVe</td>
<td>Retrieval Based</td>
<td>Multi</td>
<td>Llama 65B</td>
<td>Wikidata, QUEST, MultiSpanQA</td>
<td>Precision, F1</td>
</tr>
<tr>
<td></td>
<td>ReAct</td>
<td>Retrieval Based</td>
<td>Multi</td>
<td>PaLM-540B, GPT-3</td>
<td>HotpotQA, FEVER</td>
<td>Exact Match (EM), Accuracy</td>
</tr>
<tr>
<td>Reduce Hallucination</td>
<td>RAG</td>
<td>Retrieval Based</td>
<td>Single</td>
<td>RAG-Token, RAG-Seq.</td>
<td>MSMARCO, SearchQA</td>
<td>ROUGE, BLEU score</td>
</tr>
<tr>
<td></td>
<td>CoN</td>
<td>LM Generated</td>
<td>Multi</td>
<td>Llama 2, DPR</td>
<td>NQ, TriviaQA, WebQ</td>
<td>Exact Match (EM), F1 Score</td>
</tr>
<tr>
<td></td>
<td>CoK</td>
<td>LM Generated</td>
<td>Multi</td>
<td>gpt-3.5-turbo-0613</td>
<td>HotpotQA, FEVER, MedMCQA</td>
<td>Exact Match (EM), Accuracy</td>
</tr>
<tr>
<td>User Interaction</td>
<td>Active-Prompt</td>
<td>Manual</td>
<td>Single</td>
<td>code-davinci-002, text-davinci-003</td>
<td>Arithmetic, Commonsense, Symbolic</td>
<td>Disagreement, Entropy Variance, Self-confidence Score</td>
</tr>
<tr>
<td>Fine-Tuning and Optimization</td>
<td>APE</td>
<td>LM Generated</td>
<td>Single</td>
<td>text-curie-001, text-davanci-002</td>
<td>BBII, TruthfulQA</td>
<td>Execution accuracy, Log probability, Efficient score estimation</td>
</tr>
<tr>
<td>Knowledge-Based Reasoning and Generation</td>
<td>ART</td>
<td>Hybrid</td>
<td>Multi</td>
<td>GPT-3 (175B)</td>
<td>BigBench, MMLU</td>
<td>Accuracy</td>
</tr>
<tr>
<td>Improving Consistency and Coherence</td>
<td>CCoT</td>
<td>LM Generated</td>
<td>Multi</td>
<td>gpt-3.5-turbo-0301</td>
<td>Arithmetic, Factual QA</td>
<td>Accuracy</td>
</tr>
<tr>
<td>Managing Emotions and Tone</td>
<td>Emotion Prompting</td>
<td>Manual</td>
<td>Single</td>
<td>GPT-4</td>
<td>BIG-Bench, Instruction Induction</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>SCoT</td>
<td>Hybrid</td>
<td>Multi</td>
<td>ChatGPT, Codex</td>
<td>HumanEval, MBPP, MBCPP</td>
<td>pass@k</td>
</tr>
<tr>
<td>Code Generation and Execution</td>
<td>PoT</td>
<td>Manual</td>
<td>Single</td>
<td>gpt-3.5-turbo</td>
<td>GSM8K, SVAMP, FinQA</td>
<td>Exact Match (EM) Score</td>
</tr>
<tr>
<td></td>
<td>CoC</td>
<td>Manual</td>
<td>Single</td>
<td>text-davinci-003, gpt-3.5-turbo</td>
<td>BIG-Bench Hard</td>
<td>Accuracy</td>
</tr>
<tr>
<td></td>
<td>Scratchpad Prompting</td>
<td>Manual</td>
<td>Single</td>
<td>GPT-3</td>
<td>MBPP, MBPP-aug</td>
<td>Accuracy</td>
</tr>
<tr>
<td>Optimization and Efficiency</td>
<td>OPRO</td>
<td>Manual</td>
<td>Single</td>
<td>PaLM 2-L-IT, text-bison</td>
<td>GSM8K, BIG-Bench Hard</td>
<td>Accuracy</td>
</tr>
<tr>
<td>Understanding User Intent</td>
<td>RaR</td>
<td>Manual</td>
<td>Single</td>
<td>GPT-4-0613</td>
<td>Knowledge, Symbolic</td>
<td>Accuracy, Fair Score, Language Modeling Score</td>
</tr>
<tr>
<td>Metacognition and Self-Reflection</td>
<td>Take a Step Back</td>
<td>Manual</td>
<td>Single</td>
<td>PaLM2-L, GPT-4</td>
<td>MMLU-Physics, MMLU-Chemistry</td>
<td>Accuracy</td>
</tr>
</tbody>
</table>

# 3 Conclusion

In the domain of artificial intelligence, prompt engineering has become a transformative force, unlocking the vast potential of LLMs. This survey paper aims to serve as a foundational resource that systematically categorizes 41 distinct prompt engineering techniques based on their targeted functionalities, inspiring further research and empowering innovators in the evolving landscape of prompt engineering. The analysis spans applications, models, and datasets, shedding light on the strengths and limitations of each approach. Furthermore, we have added a diagram and a table to highlight the important points. Despite the remarkable successes, challenges persist, including biases, factual inaccuracies, and interpretability gaps, necessitating further investigation and mitigation strategies. The future of prompt engineering holds immense potential, with emerging trends like meta-learning and hybrid prompting architectures promising amplified capabilities. However, ethical considerations are paramount, emphasizing responsible development and deployment to ensure positive integration into our lives.

