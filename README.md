# HumanSD

---

This repository contains the implementation of the following paper:
> **HumanSD: A Native Skeleton-Guided Diffusion Model for Human Image Generation** [[Project Page]](https://idea-research.github.io/HumanSD/) [[Paper]](https://drive.google.com/file/d/15bbqfVZ8tF5HKpYApE2naqk12DNe14TR/view?usp=share_link) [[Code]](https://github.com/IDEA-Research/HumanSD) <br>
> [Xuan Ju](https://juxuan.space/)<sup>∗12</sup>, [Chenchen Zhao](https://zcc31415926.github.io/)<sup>∗2</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>∗1</sup>, [Jianan Wang](https://github.com/wendyjnwang/)<sup>1</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
> <sup>∗</sup> Equal contribution <sup>1</sup>International Digital Economy Academy <sup>2</sup>The Chinese University of Hong Kong


In this work, we propose a native skeleton-guided diffusion model for controllable HIG called HumanSD. Instead of performing image editing with dual-branch diffusion, we fine-tune the original SD model using a novel heatmap-guided denoising loss. This strategy effectively and efficiently strengthens the given skeleton condition during model training while mitigating the catastrophic forgetting effects. HumanSD is fine-tuned on the assembly of
three large-scale human-centric datasets with text-imagepose information, two of which are established in this work. 

---

<div  align="center">    
<img src="assets/teaser.png" width="95%">
</div>



Each group of displayed images includes: (a) a generation by the pre-trained pose-less text-guided [stable diffusion (SD)](https://github.com/Stability-AI/stablediffusion), (b) pose skeleton images as the condition to ControlNet and our proposed HumanSD, (c) a generation by [ControlNet](https://github.com/lllyasviel/ControlNet), and (d) a generation by HumanSD (ours). ControlNet and HumanSD receive both text and pose conditions. HumanSD shows its superiorities in terms of (I) challenging poses, (II) accurate painting styles, (III) pose control capability, (IV) multi-person scenarios, and (V) delicate details. 

As shown in the figure, HumanSD outperforms ControlNet in terms of accurate pose control and image quality, particularly when the given skeleton guidance is sophisticated.

## TODO

- [ ] Release inference code and pretrained models
- [ ] Release Gradio UI and Hugging Face demo
- [ ] Release training code
- [ ] Public training data (LAION-Aesthetics, HIG)

## Model Overview

<div  align="center">    
<img src="assets/model.png" width="95%">
</div>

## Quantitative Results

<div  align="center">    
<img src="assets/quantitative_results.png" width="97%">
</div>

## Qualitative Results


- (a) a generation by the pre-trained text-guided [stable diffusion (SD)](https://github.com/Stability-AI/stablediffusion)
- (b) pose skeleton images as the condition to ControlNet, T2I-Adapter and our proposed HumanSD
- (c) a generation by [ControlNet](https://github.com/lllyasviel/ControlNet)
- (d) a generation by [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)
- (e) a generation by HumanSD (ours). 

ControlNet, T2I-Adapter, and HumanSD receive both text and pose conditions.

### Natural Scene

<div  align="center">    
<img src="assets/natural1.png" width="75%">
</div>

<div  align="center">    
<img src="assets/natural3.png" width="75%">
</div>


<div  align="center">    
<img src="assets/natural2.png" width="75%">
</div>


<div  align="center">    
<img src="assets/natural4.png" width="75%">
</div>


<div  align="center">    
<img src="assets/natural5.png" width="75%">
</div>

### Sketch Scene

<div  align="center">    
<img src="assets/sketch1.png" width="75%">
</div>


<div  align="center">    
<img src="assets/sketch2.png" width="75%">
</div>

### Shadow Play Scene

<div  align="center">    
<img src="assets/shadowplay1.png" width="75%">
</div>

### Children Drawing Scene

<div  align="center">    
<img src="assets/childrendrawing1.png" width="75%">
</div>

### Oil Painting Scene

<div  align="center">    
<img src="assets/oilpainting1.png" width="75%">
</div>

### Watercolor Scene

<div  align="center">    
<img src="assets/watercolor1.png" width="75%">
</div>

### Digital Art Scene

<div  align="center">    
<img src="assets/digitalart1.png" width="75%">
</div>

### Relief Scene

<div  align="center">    
<img src="assets/relief1.png" width="75%">
</div>

### Sculpture Scene

<div  align="center">    
<img src="assets/sculpture1.png" width="75%">
</div>


