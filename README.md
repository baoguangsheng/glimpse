# Probability Distribution Estimation (PDE)
**This code is for our paper "White-Box Text Detectors Using Proprietary LLMs: A Probability Distribution Estimation Approach"**, where we borrow or extend some code from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt).

[Paper]() 
| [OnlineDemo]()


## Brief Intro
<table class="tg"  style="padding-left: 30px;">
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">ChatGPT</th>
    <th class="tg-0pky">GPT-4</th>
    <th class="tg-0pky">Claude-3<br/>Sonnet</th>
    <th class="tg-0pky">Claude-3<br/>Opus</th>
    <th class="tg-0pky">Gemini-1.5<br/>Pro</th>
    <th class="tg-0pky">Avg.</th>
  </tr>
  <tr>
    <td class="tg-0pky">Fast-DetectGPT<br/>(Open-Source: GPT-Neo-2.7B)</td>
    <td class="tg-0pky">0.9615</td>
    <td class="tg-0pky">0.9061</td>
    <td class="tg-0pky">0.9304</td>
    <td class="tg-0pky">0.9519</td>
    <td class="tg-0pky">0.8099</td>
    <td class="tg-0pky">0.9119</td>
  </tr>
  <tr>
    <td class="tg-0pky">PDE (Fast-DetectGPT)<br/>(Proprietary: GPT-3.5)</td>
    <td class="tg-0pky"><b>0.9827</b><br/>(<b>↑55%</b>)</td>
    <td class="tg-0pky"><b>0.9486</b><br/>(<b>↑45%</b>)</td>
    <td class="tg-0pky"><b>0.9638</b><br/>(<b>↑48%</b>)</td>
    <td class="tg-0pky"><b>0.9805</b><br/>(<b>↑59%</b>)</td>
    <td class="tg-0pky"><b>0.9391</b><br/>(<b>↑68%</b>)</td>
    <td class="tg-0pky"><b>0.9630</b><br/>(<b>↑58%</b>)</td>
  </tr>
</table>
The table shows detection accuracy (measured in AUROC) across five latest LLMs, where the baseline Fast-DetectGPT uses an open-source model but our PDE (Fast-DetectGPT) uses a proprietary model. The scores are averaged across the three datasets in the main results, and the notion "↑" indicates the improvement relative to the remaining space, calculated by "(new - old) / (1.0 - old)".
      
      

## Environment
* Python3.12
* PyTorch2.3.1
* Setup the environment:
  ```pip install -r requirements.txt```
  
(Notes: the baseline methods are run on 1 GPU of Tesla A100 with 80G memory, while PDE is run on a CPU environment.)

## Workspace
Following folders are created for our experiments:
* ./exp_main -> experiments with five latest LLMs as the source model (main.sh).
* ./exp_langs -> experiments on six languages (langs.sh).

(Notes: we share the data and results for convenient reproduction.)

### Citation
If you find this work useful, you can cite it with the following BibTex entry:

    @articles{bao2024white,
      title={White-Box Text Detectors Using Proprietary LLMs: A Probability Distribution Estimation Approach},
      author={Bao, Guangsheng and Zhao, Yanbin and He, Juncai and Zhang, Yue},
      year={2024}
    }

